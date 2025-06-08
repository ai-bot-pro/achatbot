# https://github.com/McGill-NLP/nano-aha-moment/blob/main/nano_r1.ipynb

import datetime
import os
import subprocess
import json
import socket
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import gc
import re
import time


import modal


APP_NAME = "nano_rl_r1_zero"
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
    .pip_install("wheel", "packaging")
    # https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    # vllm-0.8.3 does not support flashinfer>=0.2.3
    # see https://github.com/vllm-project/vllm/pull/15777
    .pip_install(
        "vllm==0.8.3",
        "flashinfer-python==0.2.2.post1",
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.6",
    )
    .pip_install(
        "deepspeed==0.16.4",
        "datasets==3.3.2",
        "accelerate==1.4.0",
    )
    .pip_install("tensorboard", "wandb")
    .env(
        {
            # Needed to stop DeepSpeed from complaining
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "6008",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": IMAGE_GPU.split(":")[-1] if len(IMAGE_GPU.split(":")) > 1 else "1",
        }
    )
    .env(
        {
            # RuntimeError: The kernel on this machine does not support the pidfd_open syscall needed to use IPC for CUDA tensors when expandable_segments:True is set. Consider using expandable_segments:False via torch.cuda.memory._set_allocator_settings('expandable_segments:False') for this allocation.
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
            "PYTHONUNBUFFERED": "1",
            "LLM_MODEL": os.getenv("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
            "DATA_PATH": os.getenv("DATA_PATH", "Jiayi-Pan/Countdown-Tasks-3to4"),
        }
    )
    .env(
        {
            # https://github.com/vllm-project/vllm/issues/16607
            "VLLM_USE_V1": "0",
        }
    )
)


with img.imports():
    from tqdm import trange
    import numpy as np
    import torch
    import wandb
    import datasets
    import deepspeed
    from deepspeed import DeepSpeedEngine
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
    from vllm import LLM, SamplingParams


HF_MODEL_DIR = "/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
DATASETS_DIR = "/datasets"
datasets_vol = modal.Volume.from_name("datasets", create_if_missing=True)
TRAIN_OUTPUT_DIR = "/train_output"
train_out_vol = modal.Volume.from_name("train_output", create_if_missing=True)

TRITON_CACHE_DIR = "/root/.triton"
triton_vol = modal.Volume.from_name("triton_cache", create_if_missing=True)


"""
# NOTE: IT model eos_token: <|im_end|>(151645) to chat complate, base model eos_token: <|endoftext|>(151643) to generate complate

# use IT chat template tokenizer
modal run src/download_models.py --repo-ids "Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-0.5B"
modal run src/download_models.py --repo-ids "Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-1.5B"
modal run src/download_models.py --repo-ids "Qwen/Qwen2.5-3B-Instruct,Qwen/Qwen2.5-3B"

modal run src/train/nano_rl/r1_zero.py
IMAGE_GPU=L40s LLM_MODEL=Qwen/Qwen2.5-1.5B modal run src/train/nano_rl/r1_zero.py --gpu-memory-utilization=0.2 --train-micro-batch-size-per-gpu=6 --train-batch-size=360
IMAGE_GPU=A100-80GB LLM_MODEL=Qwen/Qwen2.5-3B modal run src/train/nano_rl/r1_zero.py --gpu-memory-utilization=0.2 --train-micro-batch-size-per-gpu=4 --train-batch-size=64
IMAGE_GPU=A100-80GB LLM_MODEL=Qwen/Qwen2.5-3B modal run src/train/nano_rl/r1_zero.py --gpu-memory-utilization=0.2 --train-micro-batch-size-per-gpu=8 --train-batch-size=256
IMAGE_GPU=A100-80GB LLM_MODEL=Qwen/Qwen2.5-3B modal run src/train/nano_rl/r1_zero.py --gpu-memory-utilization=0.2 --train-micro-batch-size-per-gpu=8 --train-batch-size=256 --learning-rate=5e-7
"""


@app.function(
    gpu=IMAGE_GPU,
    cpu=2.0,
    secrets=[modal.Secret.from_name("achatbot")],
    retries=modal.Retries(initial_delay=0.0, max_retries=1),
    image=img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        DATASETS_DIR: datasets_vol,
        TRAIN_OUTPUT_DIR: train_out_vol,
        # TRITON_CACHE_DIR: triton_vol,
    },
    timeout=86400,  # [10,86400]
    max_containers=1,
)
def run(
    retrain: bool = False,
    # rollout
    gpu_memory_utilization: float = 0.6,
    max_response_tokens: int = 1024,
    temperature: float = 1.0,
    generations_per_sample: int = 4,  # rollout n
    # train
    num_iterations: int = 1000,
    train_batch_size: int = 64,
    train_micro_batch_size_per_gpu: int = 4,
    kl_coeff: float = 0.001,
    learning_rate: float = 1e-6,
    save_freq: int = 20,
    test_freq: int = 20,
):
    # --------------- init ----------------------
    llm_model = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    MODEL_NAME = llm_model
    MODEL_CHAT_NAME = llm_model
    if "-Instruct" in llm_model:
        MODEL_NAME = llm_model.replace("-Instruct", "")
        MODEL_CHAT_NAME = llm_model
    else:
        MODEL_NAME = llm_model
        MODEL_CHAT_NAME = llm_model + "-Instruct"

    LLM_MODEL_PATH = os.path.join(HF_MODEL_DIR, MODEL_NAME)
    LLM_CHAT_MODEL_PATH = os.path.join(HF_MODEL_DIR, MODEL_CHAT_NAME)

    OUTPUT_DIR = os.path.join(TRAIN_OUTPUT_DIR, f"{APP_NAME}")
    cur_date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    PRE_OUTPUT_DIR = os.path.join(TRAIN_OUTPUT_DIR, f"{APP_NAME}_{cur_date}")
    if retrain is True and os.path.exists(OUTPUT_DIR):
        cmd = f"mv {OUTPUT_DIR} {PRE_OUTPUT_DIR}"
        print(cmd)
        subprocess.run(cmd, shell=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --------------- run ----------------------
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    gpu_prop = torch.cuda.get_device_properties("cuda")
    print(gpu_prop)

    # RL parameters
    # Total number of training iterations
    NUM_ITERATIONS = num_iterations
    # Number of episodes to collect per iteration for training
    EPISODES_PER_ITERATION = train_batch_size
    # Number of responses to generate for each input prompt
    GENERATIONS_PER_SAMPLE = generations_per_sample
    # Controls how much the policy can deviate from the reference model
    KL_COEFFICIENT = kl_coeff

    # Training hyperparameters
    # Batch size for each GPU device during training
    PER_DEVICE_BATCH_SIZE = train_micro_batch_size_per_gpu
    # Learning rate for model updates
    LEARNING_RATE = learning_rate

    # Sampling parameters
    # Maximum number of tokens to generate in each response
    MAX_RESPONSE_TOKENS = max_response_tokens
    # Controls randomness in generation (higher = more random)
    TEMPERATURE = temperature
    # Nucleus sampling parameter (1.0 = disabled)
    TOP_P = 1.0
    # Top-k sampling parameter (-1 = disabled)
    TOP_K = -1  # no top k

    # DeepSpeed configuration
    deepspeed_config = {
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": 2, "overlap_comm": False},
        "train_batch_size": EPISODES_PER_ITERATION,
        "train_micro_batch_size_per_gpu": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation_steps": EPISODES_PER_ITERATION // PER_DEVICE_BATCH_SIZE,
        "gradient_clipping": 1.0,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": LEARNING_RATE,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.0,
                "torch_adam": True,
            },
        },
    }
    ref_deepspeed_config = {
        "bf16": {"enabled": True},
        "train_batch_size": EPISODES_PER_ITERATION,
        "train_micro_batch_size_per_gpu": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation_steps": EPISODES_PER_ITERATION // PER_DEVICE_BATCH_SIZE,
    }

    model_name_short = LLM_MODEL_PATH.split("/")[-1]
    RUN_NAME = f"{model_name_short}_temp{TEMPERATURE}_kl{KL_COEFFICIENT}_lr{LEARNING_RATE}"
    EXP_DIR = Path(os.path.join(OUTPUT_DIR, RUN_NAME))
    os.makedirs(EXP_DIR, exist_ok=True)

    print(f"Logs and Checkpoints will be saved to: {EXP_DIR}")

    ############################################
    # load Dataset
    ############################################
    data_path = os.getenv("DATA_PATH", "Jiayi-Pan/Countdown-Tasks-3to4")
    DATA_PATH = os.path.join(DATASETS_DIR, data_path)
    train_file = os.path.join(DATA_PATH, "train.parquet")
    test_file = os.path.join(DATA_PATH, "test.parquet")
    train_dataset = datasets.load_dataset("parquet", data_files=train_file)["train"]
    test_dataset = datasets.load_dataset("parquet", data_files=test_file)["train"]
    print(f"{train_dataset[0]=}")
    print(f"{test_dataset[0]=}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    ############################################
    # Initialize Models
    ############################################

    tokenizer = AutoTokenizer.from_pretrained(LLM_CHAT_MODEL_PATH)
    EOS_TOKEN_ID = AutoTokenizer.from_pretrained(LLM_MODEL_PATH).eos_token_id
    EOS_TOKEN = tokenizer.convert_ids_to_tokens(EOS_TOKEN_ID)
    print(f"chat {EOS_TOKEN=} | {EOS_TOKEN_ID=}")

    policy_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_PATH,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map=0,
    )
    reference_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_PATH,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map=0,
    )
    policy_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Initialize DeepSpeed engines
    policy_model, *_ = deepspeed.initialize(
        model=policy_model,
        config=deepspeed_config,
        model_parameters=policy_model.parameters(),
    )
    reference_model, *_ = deepspeed.initialize(
        model=reference_model,
        config=ref_deepspeed_config,
    )

    reference_model.module.cpu()

    ############################################
    # Initialize vLLM (Inference) engine
    ############################################

    # https://docs.vllm.ai/en/latest/api/vllm/config.html
    inference_engine = LLM(
        model=LLM_MODEL_PATH,
        skip_tokenizer_init=False,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True,
        swap_space=1,
        # https://docs.vllm.ai/en/latest/api/vllm/config.html?query=fcfs#__codelineno-0-2082
        scheduling_policy="fcfs",
        dtype=torch.bfloat16,
        max_model_len=2048,
        enable_sleep_mode=True,
    )

    # Wandb for logging
    wandb.init(
        project=APP_NAME,
        name=RUN_NAME,
        config={
            "model_name": MODEL_NAME,
            "learning_rate": LEARNING_RATE,
            "num_iterations": NUM_ITERATIONS,
            "episodes_per_iteration": EPISODES_PER_ITERATION,
            "rollouts_per_episode": GENERATIONS_PER_SAMPLE,
            "kl_coefficient": KL_COEFFICIENT,
            "temperature": TEMPERATURE,
        },
    )

    # Load checkpoint if it exists
    begin_iter = 0
    ckpt_path, ckpt_iter = find_last_checkpoint(EXP_DIR)
    if ckpt_path is not None:
        print(f"Resuming from checkpoint {ckpt_path} at iteration {ckpt_iter}")
        out = policy_model.load_checkpoint(ckpt_path / "deepspeed")
        if out is None:
            raise RuntimeError(f"Failed to load checkpoint {ckpt_path}")
        begin_iter = ckpt_iter + 1
        load_model_into_vllm(policy_model, inference_engine)

    for iteration in trange(begin_iter, NUM_ITERATIONS):
        print(f"Iteration {iteration}/{NUM_ITERATIONS}")

        metrics = {}

        #########################################################
        # Evaluation
        #########################################################

        eval_stats = None
        if iteration % test_freq == 0:
            print("Evaluating on eval set...")
            eval_episodes, eval_stats = evaluate_on_test_set(
                inference_engine=inference_engine,
                test_dataset=test_dataset,
                tokenizer=tokenizer,
                eos_token=EOS_TOKEN,
                eval_sampling_params=SamplingParams(
                    temperature=0.3,
                    max_tokens=1024,
                    n=1,
                    detokenize=False,
                    stop_token_ids=[EOS_TOKEN_ID],
                ),
                reward_func=lambda completion, sample: compute_reward(
                    completion, sample, EOS_TOKEN
                ),
            )
            eval_episode_table = dump_episodes(
                episodes=eval_episodes,
                episodes_stats=eval_stats,
                exp_dir=EXP_DIR,
                tokenizer=tokenizer,
                iteration=iteration,
                is_eval=True,
            )
            wandb.log({"eval/episodes": eval_episode_table, "iteration": iteration})

        #########################################################
        # Generate Episodes
        #########################################################

        # Sample training batch from 489864 train_dataset
        num_samples = EPISODES_PER_ITERATION // GENERATIONS_PER_SAMPLE
        indices = np.random.choice(len(train_dataset), size=num_samples, replace=False)
        samples = train_dataset.select(indices)

        gen_time = time.time()

        # Sample responses
        outputs = inference_engine.generate(
            prompt_token_ids=samples["input_ids"],
            sampling_params=SamplingParams(
                n=GENERATIONS_PER_SAMPLE,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                max_tokens=MAX_RESPONSE_TOKENS,
                detokenize=False,
                stop_token_ids=[EOS_TOKEN_ID],
            ),
        )
        # print(f"rollout result: {outputs[0]}")
        all_generations = [list(g.token_ids) for out in outputs for g in out.outputs]
        all_finish_reasons = [g.finish_reason for out in outputs for g in out.outputs]
        # https://docs.vllm.ai/en/latest/api/vllm/index.html?h=sleep#vllm.LLM.sleep
        # offload model to cpu and remove kv cache
        inference_engine.sleep(1)

        print(f"Generated {len(all_generations)} responses")
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        print(
            f"Time taken to generate {len(all_generations)} responses: {time.time() - gen_time} seconds"
        )

        # Process responses and calculate rewards
        episodes, episodes_stats = create_training_episodes(
            samples,
            all_generations,
            all_finish_reasons,
            tokenizer,
            EOS_TOKEN_ID,
            EOS_TOKEN,
            GENERATIONS_PER_SAMPLE,
        )
        for k, v in episodes_stats.items():
            metrics.setdefault(k, []).extend(v)

        episode_table = dump_episodes(
            episodes=episodes,
            episodes_stats=episodes_stats,
            exp_dir=EXP_DIR,
            tokenizer=tokenizer,
            iteration=iteration,
        )

        #########################################################
        # Training
        #########################################################

        # Prepare training batch
        model_inputs = prepare_model_inputs(
            query_token_ids=episodes["all_query_token_ids"],
            response_token_ids=episodes["all_response_token_ids"],
            advantages=episodes["all_advantages"],
            device="cuda",
        )

        # Calculate losses and update model
        policy_model.train()
        reference_model.module.cuda()
        reference_model.eval()

        total_response_len = (model_inputs["labels"] != -100).sum().item()

        train_time = time.time()

        for i in trange(
            0,
            EPISODES_PER_ITERATION,
            PER_DEVICE_BATCH_SIZE,
            desc="Gradient Accumulation",
        ):
            batch = {k: v[i : i + PER_DEVICE_BATCH_SIZE] for k, v in model_inputs.items()}

            # Compute policy gradient loss
            loss, loss_metrics = compute_pg_loss(
                policy_model=policy_model,
                reference_model=reference_model,
                batch=batch,
                total_response_len=total_response_len,
                TEMPERATURE=TEMPERATURE,
                KL_COEFFICIENT=KL_COEFFICIENT,
            )

            # Track metrics
            metrics.setdefault("loss", []).append(loss.item())
            grad_norm = policy_model.get_global_grad_norm()
            if grad_norm is not None:
                grad_norm = grad_norm.item()
            metrics.setdefault("grad_norm", []).append(grad_norm)
            for k, v in loss_metrics.items():
                metrics.setdefault(k, []).append(v.item() if isinstance(v, torch.Tensor) else v)

            # Backpropagation and optimization step
            policy_model.backward(loss, scale_wrt_gas=False)

            # Free memory
            del loss, loss_metrics
            if policy_model.is_gradient_accumulation_boundary():
                reference_model.module.cpu()

            policy_model.step()

        print(f"Time taken to train: {time.time() - train_time} seconds")

        #########################################################
        # Update inference engine weights
        #########################################################

        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        # load inference model to cuda  from policy model
        inference_engine.wake_up()
        load_model_into_vllm(policy_model, inference_engine)

        #########################################################
        # Log metrics
        #########################################################

        train_metrics = {k: np.mean(v) for k, v in metrics.items() if None not in v}
        train_metrics["learning_rate"] = policy_model.get_lr()[0]
        logs = {
            "iteration": iteration,
            f"episodes/iter_{iteration:06d}": episode_table,
            **{f"train/{k}": v for k, v in train_metrics.items()},
        }
        if eval_stats is not None:
            logs.update({f"eval/{k}": np.mean(v) for k, v in eval_stats.items()})
        wandb.log(logs)

        selected_keys = [
            "train/kl_penalty",
            "train/rewards",
            "train/reward_metrics/format_reward",
            "train/reward_metrics/equation_reward",
            "eval/rewards",
            "eval/reward_metrics/format_reward",
            "eval/reward_metrics/equation_reward",
        ]
        selected_metrics = {k: logs[k] for k in selected_keys if k in logs}
        print(f"KEY METRICS: {selected_metrics}")

        if iteration % save_freq == 0 and iteration != 0:
            policy_model.module.save_pretrained(
                str(EXP_DIR / "checkpoints" / f"ckpt_{iteration:06d}" / "hf_model")
            )
            policy_model.save_checkpoint(
                str(EXP_DIR / "checkpoints" / f"ckpt_{iteration:06d}" / "deepspeed")
            )


def prepare_model_inputs(
    query_token_ids: List[List[int]],
    response_token_ids: List[List[int]],
    advantages: List[List[float]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Prepare padded model inputs with attention masks, labels, and advantages.
    Args:
        query_token_ids: List of query token ids
        response_token_ids: List of response token ids
        advantages: List of lists of advantage values, matching response_token_ids structure
        device: Device to move the tensors to
    Returns:
        Dict with input_ids, attention_mask, labels, and advantages

    Example:
        >>> query_token_ids = [[1, 2, 3], [4, 5]]
        >>> response_token_ids = [[6, 7], [8]]
        >>> advantages = [[0.5, 0.8], [0.3]]
        >>> outputs = prepare_model_inputs(query_token_ids, response_token_ids, advantages, "cuda")
        >>> outputs
        {
            'input_ids': tensor([
                [1, 2, 3, 6, 7],
                [4, 5, 8, 0, 0]
            ]),
            'attention_mask': tensor([
                [1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0]
            ]),
            'labels': tensor([
                [-100, -100, -100, 6, 7],
                [-100, -100, 8, -100, -100]
            ]),
            'advantages': tensor([
                [0.0, 0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.0, 0.9, 0.0]
            ])
        }
    """
    max_seq_len = max(len(q) + len(r) for q, r in zip(query_token_ids, response_token_ids))
    inputs = {"input_ids": [], "attention_mask": [], "labels": [], "advantages": []}

    pad_token_id = 0  # Doesn't matter, will be masked
    ignore_index = -100

    for query, response, advantage in zip(query_token_ids, response_token_ids, advantages):
        combined_ids = query + response
        seq_len = len(combined_ids)

        # Create padded sequences
        input_ids = combined_ids + [pad_token_id] * (max_seq_len - seq_len)
        attention_mask = [1] * seq_len + [0] * (max_seq_len - seq_len)
        labels = [ignore_index] * len(query) + response + [ignore_index] * (max_seq_len - seq_len)
        advantages_seq = [0.0] * len(query) + advantage + [0.0] * (max_seq_len - seq_len)

        assert len(input_ids) == max_seq_len
        assert len(attention_mask) == max_seq_len
        assert len(labels) == max_seq_len
        assert len(advantages_seq) == max_seq_len

        inputs["input_ids"].append(input_ids)
        inputs["attention_mask"].append(attention_mask)
        inputs["labels"].append(labels)
        inputs["advantages"].append(advantages_seq)

    # Convert to tensors
    return {
        k: torch.tensor(v, dtype=torch.long if k != "advantages" else torch.float, device=device)
        for k, v in inputs.items()
    }


def compute_token_log_probs(
    model: Union["DeepSpeedEngine", "PreTrainedModel"],
    inputs: Dict[str, torch.Tensor],
    temperature: float,
) -> torch.Tensor:
    """
    Compute log probabilities for each token in the sequence, masked for valid labels only.

    This function:
    1. Runs the model forward pass
    2. Applies temperature scaling to logits
    3. Shifts the sequences for causal language modeling
    4. Computes log probabilities for the actual tokens that appeared in the sequence
    5. Masks the log probabilities to only include valid labels (non -100 positions)

    Args:
        model: The language model (either DeepSpeed-wrapped or regular HuggingFace model)
        inputs: Dictionary containing:
            - input_ids: Tensor of token ids [batch_size, seq_len]
            - attention_mask: Tensor of attention mask [batch_size, seq_len]
            - labels: Tensor of target labels [batch_size, seq_len] with -100 for ignored positions
        temperature: Temperature for scaling the logits before softmax

    Returns:
        torch.Tensor: Log probabilities tensor of shape [batch_size, seq_len-1], where:
            - Each value is the log probability of the actual token that appeared
            - Values are masked to 0.0 for positions where labels were -100
            - The sequence length is reduced by 1 due to the causal shift

    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> inputs = {
        ...     "input_ids": torch.tensor([[1, 2, 3]]),
        ...     "attention_mask": torch.tensor([[1, 1, 1]]),
        ...     "labels": torch.tensor([[-100, 2, 3]])
        ... }
        >>> log_probs = compute_token_log_probs(model, inputs, temperature=1.0)
        >>> log_probs.shape
        torch.Size([1, 2])  # batch_size=1, seq_len-1=2
        >>> # First position is 0 (masked), second position has actual log prob
    """
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        return_dict=True,
        use_cache=False,
    )

    logits = outputs.logits.float() / temperature  # Shape: [batch_size, seq_len, vocab_size]
    shift_logits = logits[..., :-1, :].contiguous()  # Shape: [batch_size, seq_len-1, vocab_size]
    shift_labels = inputs["labels"][..., 1:].contiguous()  # Shape: [batch_size, seq_len-1]

    # Create mask for valid labels
    label_mask = (shift_labels != -100).float()  # Shape: [batch_size, seq_len-1]
    shift_labels[shift_labels == -100] = 0  # Shape: [batch_size, seq_len-1]

    # Calculate log probabilities
    log_probs = torch.log_softmax(
        shift_logits, dim=-1
    )  # Shape: [batch_size, seq_len-1, vocab_size]
    log_probs = torch.gather(
        log_probs, dim=2, index=shift_labels.unsqueeze(2)
    )  # Shape: [batch_size, seq_len-1, 1]
    log_probs = log_probs.squeeze(2)  # Shape: [batch_size, seq_len-1]
    log_probs = log_probs * label_mask  # Shape: [batch_size, seq_len-1]

    return log_probs


def find_free_port():
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def evaluate_on_test_set(
    inference_engine: "LLM",
    test_dataset: "datasets.Dataset",
    tokenizer: "AutoTokenizer",
    eos_token: str,
    eval_sampling_params: "SamplingParams",
    reward_func: Callable[[str, Dict[str, Any]], Tuple[float, Dict[str, float]]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate the model on a test dataset by generating responses and computing rewards.

    Args:
        inference_engine: The sglang Engine instance used for text generation
        test_dataset: Dataset containing test samples
        tokenizer: Tokenizer for decoding generated token IDs
        eos_token: End of sequence token string
        eval_sampling_params: Dictionary of parameters for controlling the generation process
        reward_func: Function that computes rewards for generated responses. Takes a response
            string and sample dict as input, returns a tuple of (overall_reward, reward_components)

    Returns:
        Dictionary containing evaluation statistics:
            - response_lengths: List of token counts for each generated response
            - rewards: List of overall reward values for each response
            - non_stop_rate: List of booleans indicating if generation ended for non-stop reason
            - reward_metrics/*: Lists of individual reward component values, prefixed with
              "reward_metrics/"
        episodes: Dictionary containing:
            - all_query_token_ids: List of query token IDs for each episode
            - all_response_token_ids: List of response token IDs for each episode

    Example:
        >>> episodes, episodes_stats = evaluate_on_test_set(
        ...     inference_engine=engine,
        ...     test_dataset=dataset,
        ...     tokenizer=tokenizer,
        ...     eos_token="</s>",
        ...     eval_sampling_params={"temperature": 0.7, "max_tokens": 100},
        ...     reward_func=compute_rewards
        ... )
        >>> print(f"Average reward: {episodes_stats['rewards']:.3f}")
    """
    generations = inference_engine.generate(
        prompt_token_ids=test_dataset["input_ids"], sampling_params=eval_sampling_params
    )

    metrics = {
        "response_lengths": [],
        "rewards": [],
        "non_stop_rate": [],
    }

    all_query_token_ids = []
    all_responses_token_ids = []

    for i, sample in enumerate(test_dataset):
        query_token_ids = sample["input_ids"]
        response_token_ids = generations[i].outputs[0].token_ids
        finish_reason = generations[i].outputs[0].finish_reason

        response = tokenizer.decode(response_token_ids, skip_special_tokens=False)
        reward, reward_components = reward_func(response, sample)

        all_query_token_ids.append(query_token_ids)
        all_responses_token_ids.append(response_token_ids)

        metrics["rewards"].append(reward)
        metrics["non_stop_rate"].append(finish_reason != "stop")
        metrics["response_lengths"].append(len(response_token_ids))
        for k, v in reward_components.items():
            metrics.setdefault(f"reward_metrics/{k}", []).append(v)

    episodes = {
        "all_query_token_ids": all_query_token_ids,
        "all_response_token_ids": all_responses_token_ids,
    }

    return episodes, metrics


def dump_episodes(
    episodes: Dict[str, Any],
    episodes_stats: Dict[str, Any],
    exp_dir: Path,
    tokenizer: "AutoTokenizer",
    iteration: int,
    is_eval: bool = False,
) -> "wandb.Table":
    query_token_ids = episodes["all_query_token_ids"]
    response_token_ids = episodes["all_response_token_ids"]
    rewards = episodes_stats["rewards"]
    response_lengths = episodes_stats["response_lengths"]

    query_texts = tokenizer.batch_decode(
        query_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    response_texts = tokenizer.batch_decode(
        response_token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    if not is_eval:
        print(
            f"########## Example 1 (Reward: {rewards[0]}, Response Length: {response_lengths[0]})"
        )
        print(f"#### Query:\n`{query_texts[0]}`")
        print(f"#### Response:\n`{response_texts[0]}`\n\n")

        print(
            f"########## Example 2 (Reward: {rewards[1]}, Response Length: {response_lengths[1]})"
        )
        print(f"#### Query:\n`{query_texts[1]}`")
        print(f"#### Response:\n`{response_texts[1]}`\n\n")

    if is_eval:
        episodes_dir = exp_dir / "eval_episodes"
    else:
        episodes_dir = exp_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    with open(episodes_dir / f"eps_{iteration:06d}.json", "w") as f:
        json.dump(
            [
                {
                    "query": query_texts[i],
                    "response": response_texts[i],
                    "reward": rewards[i],
                }
                for i in range(len(query_texts))
            ],
            f,
        )

    # Create wandb table
    table = wandb.Table(columns=["query", "response", "reward", "response_length"])
    for i in range(len(query_texts)):
        table.add_data(query_texts[i], response_texts[i], rewards[i], response_lengths[i])

    return table


def find_last_checkpoint(exp_dir: Path) -> Tuple[Optional[Path], Optional[int]]:
    checkpoint_dir = exp_dir / "checkpoints"
    checkpoints = list(checkpoint_dir.glob("ckpt_*"))
    # Filter out directories that don't have a deepspeed subdirectory
    checkpoints = [ckpt for ckpt in checkpoints if (ckpt / "deepspeed").exists()]
    if not checkpoints:
        return None, None
    ckpt_path = max(checkpoints, key=lambda x: int(x.stem.split("_")[-1]))
    ckpt_iter = int(ckpt_path.stem.split("_")[-1])
    return ckpt_path, ckpt_iter


def load_model_into_vllm(model: Union["DeepSpeedEngine", "PreTrainedModel"], llm: "LLM") -> None:
    """
    Load weights from a HuggingFace model (either wrapped in DeepSpeed or not) into a vLLM inference engine.

    This function transfers the weights from a training model to a vLLM inference engine,
    allowing for efficient inference using the updated model weights.

    Args:
        model (Union[DeepSpeedEngine, PreTrainedModel]): The source model to copy weights from.
            Can be either a DeepSpeed-wrapped model or a regular HuggingFace PreTrainedModel.
        vllm (LLM): The target vLLM inference engine to load the weights into.
            Must be already initialized and ready to accept new weights.

    Returns:
        None
    """
    state_dict = (
        model.module.state_dict() if isinstance(model, DeepSpeedEngine) else model.state_dict()
    )
    llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(state_dict.items())



def format_reward_func(completion: str, EOS_TOKEN: str) -> float:
    """
    Format: <think>...</think><answer>...</answer>

    Also checks that the content within <answer>...</answer> conforms to a
    specified pattern (only digits, + - * / ( ) . and whitespace).

    Args:
        completion (str): Generated output
        EOS_TOKEN (str): End of sequence token

    Returns:
        float: Reward score
    """
    # Define the allowed pattern (only numbers, +, -, *, /, (, ), ., and whitespace)
    allowed_pattern = r"^[\d+\-*/().\s]+$"

    try:
        # Synthetically prepend <think> (if your pipeline relies on that to ease matching)
        completion = "<think>" + completion

        # Strip EOS token if present
        if completion.endswith(EOS_TOKEN):
            completion = completion[: -len(EOS_TOKEN)]

        # Check if the format is correct
        # Pattern means:
        # 1) <think>...contents not including other <think> tags...</think>
        # 2) \n
        # 3) <answer>...anything...</answer>
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
        match = re.search(regex, completion, re.DOTALL)

        if match is None or len(match.groups()) != 2:
            # Format is incorrect
            return 0.0
        else:
            # Extract the content inside <answer>...</answer>
            answer_content = match.group(2).strip()

            # Check if answer content matches the allowed pattern
            if not re.match(allowed_pattern, answer_content):
                # If it doesn't match, reward is 0.5
                return 0.5
            else:
                # If both format and pattern are correct, reward is 1
                return 1.0
    except Exception:
        # Any error leads to 0 reward
        return 0.0


def equation_reward_func(completion: str, nums: List[int], target: int) -> float:
    """
    Evaluates completion based on mathematical correctness of the answer

    Args:
        completion (str): Generated output
        target (str): Expected answer
        nums (list): Available numbers to use in the equation

    Returns:
        float: Reward score
    """
    try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion
        # Check if the format is correct
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match is None:
            return 0.0
        # Extract the "answer" part from the completion
        equation = match.group(1).strip()
        # Extract all numbers from the equation
        used_numbers = [int(n) for n in re.findall(r"\d+", equation)]

        # Check if all numbers are used exactly once
        if sorted(used_numbers) != sorted(nums):
            return 0.0
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation):
            return 0.0

        # Evaluate the equation with restricted globals and locals
        result = eval(equation, {"__builtins__": None}, {})
        # Check if the equation is correct and matches the ground truth
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
        else:
            return 0.0
    except Exception:
        # If evaluation fails, reward is 0
        return 0.0


def compute_reward(
    completion: str, sample: Dict[str, Any], EOS_TOKEN: str
) -> Tuple[float, Dict[str, float]]:
    nums = sample["nums"]
    target = sample["target"]

    format_reward = format_reward_func(completion, EOS_TOKEN)
    equation_reward = equation_reward_func(completion=completion, nums=nums, target=target)

    reward = format_reward + equation_reward

    metrics = {
        "format_reward": format_reward,
        "equation_reward": equation_reward,
    }

    return reward, metrics


def create_training_episodes(
    samples: List[Dict[str, Any]],
    all_generations: List[List[int]],
    all_finish_reasons: List[str],
    tokenizer: "AutoTokenizer",
    EOS_TOKEN_ID: int,
    EOS_TOKEN: str,
    GENERATIONS_PER_SAMPLE: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Process model generations and calculate rewards for training episodes.

    This function processes generated responses and calculates rewards for training episodes by:
    1. Grouping generations by sample (GENERATIONS_PER_SAMPLE responses per input)
    2. Computing rewards and advantages for each response
    3. Processing response tokens (adding EOS tokens where needed)

    Args:
        samples: List of input samples, each containing:
            - input_ids: List[int], tokenized input prompt
            - nums: List[int], numbers to use in equation
            - target: int, target value for equation
        all_generations: List of token ID sequences for each generated response
        all_finish_reasons: List of finish reasons for each generation ("stop" or other)

    Returns:
        Tuple containing:
        1. Dictionary with processed data for training:
            - all_query_token_ids: List[List[int]], input token IDs repeated for each generation
            - all_response_token_ids: List[List[int]], response token IDs with EOS tokens added
            - all_advantages: List[List[float]], advantage values repeated for each token
        2. Dictionary with generation statistics:
            - response_lengths: List[int], lengths of generated responses
            - rewards: List[float], raw reward values
            - non_stop_rate: List[bool], whether each generation ended naturally
            - reward_metrics/*: Various reward component metrics

    Example:
        >>> samples = [{"input_ids": [1,2,3], "nums": [1,2,3], "target": 6}]
        >>> generations = [[4,5], [6,7], [8,9]]  # 3 generations per sample
        >>> finish_reasons = ["stop", "length", "stop"]
        >>> episodes, stats = create_training_episodes(samples, generations, finish_reasons)
        >>> episodes
        {
            'all_query_token_ids': [[1,2,3], [1,2,3], [1,2,3]],
            'all_response_token_ids': [[4,5,EOS], [6,7], [8,9,EOS]],
            'all_advantages': [[0.5,0.5,0.5], [-1.0,-1.0], [0.5,0.5,0.5]]
        }
    """
    assert len(all_generations) == len(all_finish_reasons)
    assert len(all_generations) == len(samples) * GENERATIONS_PER_SAMPLE

    # Process responses and calculate rewards
    groups = [
        list(range(i, i + GENERATIONS_PER_SAMPLE))
        for i in range(0, len(all_generations), GENERATIONS_PER_SAMPLE)
    ]  # example: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    all_query_token_ids, all_responses_token_ids, all_advantages = [], [], []

    stats = {
        "response_lengths": [],
        "rewards": [],
        "non_stop_rate": [],
    }

    for sample, group_indices in zip(samples, groups):
        response_token_ids = [all_generations[i] for i in group_indices]
        finish_reasons = [all_finish_reasons[i] for i in group_indices]
        responses = tokenizer.batch_decode(response_token_ids, skip_special_tokens=False)
        rewards_and_metrics = [compute_reward(resp, sample, EOS_TOKEN) for resp in responses]
        rewards, reward_metrics = zip(*rewards_and_metrics)

        rewards = np.array(rewards)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

        per_token_advantages = [
            [adv] * len(resp) for adv, resp in zip(advantages, response_token_ids)
        ]

        all_query_token_ids.extend([sample["input_ids"]] * GENERATIONS_PER_SAMPLE)
        all_responses_token_ids.extend(response_token_ids)
        all_advantages.extend(per_token_advantages)

        stats["rewards"].extend(rewards)
        stats["non_stop_rate"].extend([fr != "stop" for fr in finish_reasons])
        stats["response_lengths"].extend([len(ids) for ids in response_token_ids])
        for rm in reward_metrics:
            for k, v in rm.items():
                stats.setdefault(f"reward_metrics/{k}", []).append(v)

    episodes = {
        "all_query_token_ids": all_query_token_ids,
        "all_response_token_ids": all_responses_token_ids,
        "all_advantages": all_advantages,
    }

    return episodes, stats


def compute_pg_loss(
    policy_model: Union["DeepSpeedEngine", "PreTrainedModel"],
    reference_model: Union["DeepSpeedEngine", "PreTrainedModel"],
    batch: Dict[str, torch.Tensor],
    total_response_len: int,
    TEMPERATURE: float,
    KL_COEFFICIENT: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the policy gradient loss with KL penalty between policy and reference models.

    This function:
    1. Computes log probabilities for both policy and reference models
    2. Calculates KL divergence penalty between the models
    3. Computes policy gradient loss using advantages
    4. Combines the losses with KL coefficient

    Args:
        policy_model: The model being trained
        reference_model: The reference model for KL penalty calculation
        batch: Dictionary containing:
            - input_ids: Tensor of shape [batch_size, seq_len]
            - attention_mask: Tensor of shape [batch_size, seq_len]
            - labels: Tensor of shape [batch_size, seq_len] with -100 for ignored positions
            - advantages: Tensor of shape [batch_size, seq_len]

    Returns:
        Tuple containing:
            - loss: Combined policy gradient and KL penalty loss (scalar tensor)
            - metrics: Dictionary with detailed loss components:
                - policy_loss: Pure policy gradient loss
                - kl_penalty: KL divergence penalty
                - entropy: Policy entropy
    """
    input_ids = batch["input_ids"]  # [batch_size, seq_len]
    attention_mask = batch["attention_mask"]  # [batch_size, seq_len]
    labels = batch["labels"]  # [batch_size, seq_len]
    advantages = batch["advantages"]  # [batch_size, seq_len]

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    labels_mask = (labels[..., 1:] != -100).float()  # [batch_size, seq_len-1]

    with torch.no_grad():
        ref_logps = compute_token_log_probs(
            reference_model, model_inputs, TEMPERATURE
        )  # [batch_size, seq_len-1]

    logps = compute_token_log_probs(
        policy_model, model_inputs, TEMPERATURE
    )  # [batch_size, seq_len-1]

    kl_penalty = torch.exp(ref_logps - logps) - (ref_logps - logps) - 1  # [batch_size, seq_len-1]
    kl_penalty = kl_penalty * labels_mask  # [batch_size, seq_len-1]

    entropy = -logps.sum() / labels_mask.sum()  # scalar

    policy_loss = -logps * advantages[..., 1:]  # [batch_size, seq_len-1]
    policy_loss = policy_loss * labels_mask  # [batch_size, seq_len-1]

    loss = (policy_loss + KL_COEFFICIENT * kl_penalty).sum() / total_response_len  # scalar

    metrics = {
        "policy_loss": policy_loss.sum().item() / total_response_len,
        "kl_penalty": kl_penalty.sum().item() / total_response_len,
        "entropy": entropy.item() / total_response_len,
    }

    return loss, metrics
