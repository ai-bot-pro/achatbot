from dataclasses import dataclass
from threading import Thread
import datetime
import os
import subprocess
import random
import logging
from typing import Union
import re


import modal


APP_NAME = "trl_r1_zero"
app = modal.App(APP_NAME)
IMAGE_GPU = os.getenv("IMAGE_GPU", "L4")

img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "clang", "cmake", "ninja-build")
    .pip_install(
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "torchvision==0.21.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install("wheel", "packaging", "setuptools")
    # https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    # vllm-0.8.3 does not support flashinfer>=0.2.3
    # see https://github.com/vllm-project/vllm/pull/15777
    .pip_install(
        "vllm==0.8.3",
        "flashinfer-python==0.2.2.post1",
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.6",
    )
    .pip_install("deepspeed==0.16.4", "datasets==3.3.2", "accelerate==1.4.0", "trl==0.18.0")
    .pip_install("tensorboard", "wandb")
    .env(
        {
            # Needed to stop DeepSpeed from complaining
            # "MASTER_ADDR": "localhost",
            # "MASTER_PORT": "6008",
            # "RANK": "0",
            # "LOCAL_RANK": "0",
            # "WORLD_SIZE": IMAGE_GPU.split(":")[-1] if len(IMAGE_GPU.split(":")) > 1 else "1",
        }
    )
    .env(
        {
            # RuntimeError: The kernel on this machine does not support the pidfd_open syscall needed to use IPC for CUDA tensors when expandable_segments:True is set. Consider using expandable_segments:False via torch.cuda.memory._set_allocator_settings('expandable_segments:False') for this allocation.
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
            "PYTHONUNBUFFERED": "1",
            "LLM_MODEL": os.getenv("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        }
    )
    .env(
        {
            # https://github.com/vllm-project/vllm/issues/16607
            "VLLM_USE_V1": "0",
        }
    )
)


# Custom dataclasses
@dataclass
class ScriptArguments:
    dataset_id_or_path: str = "Jiayi-Pan/Countdown-Tasks-3to4"
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None


with img.imports():
    import torch

    import datasets
    from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset

    from transformers import AutoTokenizer
    from transformers import AutoModelForCausalLM, AutoProcessor, TextIteratorStreamer
    from transformers.trainer_utils import get_last_checkpoint

    from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)


HF_MODEL_DIR = "/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
DATASETS_DIR = "/datasets"
datasets_vol = modal.Volume.from_name("datasets", create_if_missing=True)
TRAIN_OUTPUT_DIR = "/train_output"
train_out_vol = modal.Volume.from_name("train_output", create_if_missing=True)

TRITON_CACHE_DIR = "/root/.triton"
triton_vol = modal.Volume.from_name("triton_cache", create_if_missing=True)


REMOTE_CONFIG_DIR = "/root/configs"
cur_dir = os.path.dirname(os.path.abspath(__file__))
img = img.add_local_dir(f"{cur_dir}/configs", REMOTE_CONFIG_DIR)
# https://modal.com/docs/guide/modal-1-0-migration#requiring-explicit-inclusion-of-local-python-dependencies
img = img.add_local_python_source("_remote_module_non_scriptable")


"""
# use IT chat template tokenizer
modal run src/download_models.py --repo-ids "Qwen/Qwen2.5-0.5B-Instruct"
modal run src/download_models.py --repo-ids "Qwen/Qwen2.5-1.5B-Instruct"
modal run src/download_models.py --repo-ids "Qwen/Qwen2.5-3B-Instruct"

# display config
modal run src/train/trl/r1_zero.py --display
LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct modal run src/train/trl/r1_zero.py --display
LLM_MODEL=Qwen/Qwen2.5-3B-Instruct modal run src/train/trl/r1_zero.py --display

# train
modal run src/train/trl/r1_zero.py
IMAGE_GPU=L40s LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct modal run src/train/trl/r1_zero.py
IMAGE_GPU=A100-80GB LLM_MODEL=Qwen/Qwen2.5-3B-Instruct modal run src/train/trl/r1_zero.py

# generate
modal run src/train/trl/r1_zero.py --gen
LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct modal run src/train/trl/r1_zero.py --gen
LLM_MODEL=Qwen/Qwen2.5-3B-Instruct modal run src/train/trl/r1_zero.py --gen
"""


@app.function(
    gpu=IMAGE_GPU,
    cpu=2.0,
    image=img,
    secrets=[modal.Secret.from_name("achatbot")],
    retries=modal.Retries(initial_delay=0.0, max_retries=1),
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        DATASETS_DIR: datasets_vol,
        TRAIN_OUTPUT_DIR: train_out_vol,
        # TRITON_CACHE_DIR: triton_vol,
    },
    timeout=86400,  # [10,86400]
    max_containers=1,
)
def train(
    retrain: bool = False,
    algorithm: str = "grpo",
    display: bool = False,
    gen: bool = False,
):
    # Parse config
    llm_path = os.getenv("LLM_MODEL")
    assert llm_path
    llm_name = llm_path.split("/")[-1]
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, grpo_training_args = parser.parse_args_and_config(
        args=["--config", f"{REMOTE_CONFIG_DIR}/{llm_name}-{algorithm}.yaml"]
    )
    model_args.model_name_or_path = os.path.join(HF_MODEL_DIR, model_args.model_name_or_path)

    OUTPUT_DIR = os.path.join(TRAIN_OUTPUT_DIR, f"{APP_NAME}")
    cur_date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    PRE_OUTPUT_DIR = os.path.join(TRAIN_OUTPUT_DIR, f"{APP_NAME}_{cur_date}")
    if retrain is True and os.path.exists(OUTPUT_DIR):
        cmd = f"mv {OUTPUT_DIR} {PRE_OUTPUT_DIR}"
        print(cmd)
        subprocess.run(cmd, shell=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data_name = script_args.dataset_id_or_path.split("/")[-1]
    RUN_NAME = f"{llm_name}_{algorithm}_{data_name}"
    EXP_DIR = os.path.join(OUTPUT_DIR, RUN_NAME)

    os.makedirs(EXP_DIR, exist_ok=True)
    print(f"Checkpoints will be saved to: {EXP_DIR}")
    grpo_training_args.output_dir = EXP_DIR

    LOGS_DIR = os.path.join(EXP_DIR, "logs")
    os.makedirs(LOGS_DIR, exist_ok=True)
    grpo_training_args.logging_dir = LOGS_DIR
    print(f"Logs will be saved to: {LOGS_DIR}")

    print(f"{model_args=}")
    print(f"{script_args=}")
    print(f"{grpo_training_args=}")
    if display is True:
        return

    # Load preprocessed datasets
    data_id = script_args.dataset_id_or_path
    DATA_PATH = os.path.join(DATASETS_DIR, data_id)
    train_file = os.path.join(DATA_PATH, "trl_train.parquet")
    test_file = os.path.join(DATA_PATH, "trl_test.parquet")
    train_dataset = datasets.load_dataset("parquet", data_files=train_file)[
        script_args.dataset_splits
    ]
    test_dataset = datasets.load_dataset("parquet", data_files=test_file)[
        script_args.dataset_splits
    ]
    print(f"{train_dataset[0]=}")
    print(f"{test_dataset[0]=}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    if gen is True:
        generate(EXP_DIR, test_dataset)
        return

    # Training
    print(f"start train with {algorithm}")
    if algorithm == "grpo":
        return train_grpo(
            model_args,
            script_args,
            grpo_training_args,
            train_dataset,
            test_dataset,
        )
    else:
        print(f"{algorithm} is not supported")


def format_reward_func(completions, target, **kwargs):
    """
    Format: <think>...</think><answer>...</answer>
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers

      Returns:
          list[float]: Reward scores
    """
    rewards = []

    for completion, gt in zip(completions, target):
        try:
            # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
            completion = "<think>" + completion
            if random.random() < 0.1:  # 1% chance to write samples into a file
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join("completion_samples", "completion_samples.txt")
                with open(log_file, "a") as f:
                    f.write(f"\n\n==============\n")
                    f.write(completion)

            # Check if the format is correct
            regex = (
                r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
            )

            match = re.search(regex, completion, re.DOTALL)
            # if the format is not correct, reward is 0
            if match is None or len(match.groups()) != 2:
                rewards.append(0.0)
            else:
                rewards.append(1.0)
        except Exception:
            rewards.append(0.0)
    return rewards


def equation_reward_func(completions, target, nums, **kwargs):
    """
    Evaluates completions based on:
    2. Mathematical correctness of the answer

    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
        nums (list[str]): Available numbers

    Returns:
        list[float]: Reward scores
    """
    rewards = []
    for completion, gt, numbers in zip(completions, target, nums):
        try:
            # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
            completion = "<think>" + completion
            # Check if the format is correct
            match = re.search(r"<answer>(.*?)<\/answer>", completion)
            if match is None:
                rewards.append(0.0)
                continue
            # Extract the "answer" part from the completion
            equation = match.group(1).strip()
            # Extract all numbers from the equation
            used_numbers = [int(n) for n in re.findall(r"\d+", equation)]

            # Check if all numbers are used exactly once
            if sorted(used_numbers) != sorted(numbers):
                rewards.append(0.0)
                continue
            # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
            allowed_pattern = r"^[\d+\-*/().\s]+$"
            if not re.match(allowed_pattern, equation):
                rewards.append(0.0)
                continue

            # Evaluate the equation with restricted globals and locals
            result = eval(equation, {"__builtins__": None}, {})
            # Check if the equation is correct and matches the ground truth
            if abs(float(result) - float(gt)) < 1e-5:
                rewards.append(1.0)
                if (
                    random.random() < 0.10
                ):  # 10% chance to write fully successful samples into a file
                    os.makedirs("completion_samples", exist_ok=True)
                    log_file = os.path.join("completion_samples", "success_completion_samples.txt")
                    with open(log_file, "a") as f:
                        f.write(f"\n\n==============\n")
                        f.write(completion)
            else:
                rewards.append(0.0)
        except Exception:
            # If evaluation fails, reward is 0
            rewards.append(0.0)
    return rewards


def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def train_grpo(
    model_args: ModelConfig,
    script_args: ScriptArguments,
    training_args: GRPOConfig,
    train_dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
    test_dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
):
    # Log parameters
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Instantiate GRPO trainer
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=[format_reward_func, equation_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=get_peft_config(model_args),
    )

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Train the model
    logger.info(
        f"*** Starting training {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} for {training_args.num_train_epochs} epochs***"
    )
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    # Save model and create model card
    logger.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["rl", "grpo", "tutorial"]})
    # push to hub if needed
    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

    logger.info("*** Training complete! ***")


@torch.inference_mode()
def generate(
    model_path: str,
    test_dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
):
    gpu_prop = torch.cuda.get_device_properties("cuda")
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
            trust_remote_code=True,
        )
        .to(torch.device("cuda"))
        .eval()
    )
    tokenizer = AutoProcessor.from_pretrained(model_path, use_fast=True)

    SYSTEM_MESSAGE = (
        "You are a helpful assistant. You first think about the reasoning process in the mind "
        "and then provide the user with the answer."
    )
    PROMPT_TEMPLATE = (
        "Using the numbers {numbers}, create an equation that equals {target}. "
        "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
        "Show your work in <think> </think> tags. And return the final equation and answer in "
        "<answer> </answer> tags, for example <answer>(1 + 2) / 3 = 1</answer>."
    )

    numbers = [95, 21, 3]
    target = 88
    msgs = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(numbers=numbers, target=target),
        },
    ]
    inputs = tokenizer.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    print(inputs)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=False, skip_special_tokens=False)
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        do_sample=True,
        temperature=0.6,
        top_k=1,
        top_p=None,
        repetition_penalty=1.1,
        min_new_tokens=0,
        max_new_tokens=128,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        print(new_text, flush=True, end="")
    print(f"\n{generated_text=}")

    torch.cuda.empty_cache()
