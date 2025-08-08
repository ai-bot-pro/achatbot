import os
import asyncio
import subprocess


import modal


app = modal.App("openai_gpt_oss_trl")
IMAGE_GPU = os.getenv("IMAGE_GPU", None)
img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvcr.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs")
    .pip_install(
        "trl>=0.20.0",
        "peft>=0.17.0",
        "transformers>=4.55.0",  # NOTE: update to >4.55.0 to support T4 inference with mxfp4
        "trackio",
    )
    .pip_install("accelerate", "kernels", "triton==3.4.0")
    .run_commands(
        "pip install -U git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels",
    )
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            # "TQDM_DISABLE": "1",
            "LLM_MODEL": os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
        }
    )
)


HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
TRAIN_OUTPUT_DIR = "/train_output"
train_out_vol = modal.Volume.from_name("train_output", create_if_missing=True)
HUGGINGFACE_CACHE_DIR = "/root/.cache/huggingface/"
huggingface_cache_vol = modal.Volume.from_name("huggingface_cache", create_if_missing=True)


with img.imports():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, PeftModel

    from trl import SFTConfig, SFTTrainer

    MODEL_PATH = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    model_path = os.path.join(HF_MODEL_DIR, MODEL_PATH)


@app.function(
    gpu=IMAGE_GPU,
    cpu=2.0,
    retries=1,
    image=img,
    secrets=[modal.Secret.from_name("achatbot")],
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        TRAIN_OUTPUT_DIR: train_out_vol,
        HUGGINGFACE_CACHE_DIR: huggingface_cache_vol,
    },
    timeout=86400,  # default 300s
    max_containers=1,
)
async def run(func, **kwargs):
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    if torch.cuda.is_available():
        gpu_prop = torch.cuda.get_device_properties("cuda")
        print(gpu_prop)

    if asyncio.iscoroutinefunction(func):
        await func(**kwargs)
    else:
        func(**kwargs)


def train(**kwargs):
    dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")
    print(dataset, dataset[0])
    messages = dataset[0]["messages"]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    conversation = tokenizer.apply_chat_template(messages, tokenize=False)
    print(conversation)

    quantization_config = Mxfp4Config(dequantize=True)
    print(quantization_config)
    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        use_cache=False,
        device_map="auto",
    )

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    messages = [
        {"role": "user", "content": "¿Cuál es el capital de Australia?"},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(input_ids, max_new_tokens=512)
    response = tokenizer.batch_decode(output_ids)[0]
    print(response)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules="all-linear",
        target_parameters=[
            "7.mlp.experts.gate_up_proj",
            "7.mlp.experts.down_proj",
            "15.mlp.experts.gate_up_proj",
            "15.mlp.experts.down_proj",
            "23.mlp.experts.gate_up_proj",
            "23.mlp.experts.down_proj",
        ],
    )
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    training_args = SFTConfig(
        learning_rate=2e-4,
        gradient_checkpointing=True,
        num_train_epochs=1,
        logging_steps=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_length=2048,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},
        output_dir=os.path.join(
            TRAIN_OUTPUT_DIR, f"{MODEL_PATH.split('/')[-1]}-trl-multilingual-reasoner"
        ),
        report_to="trackio",
        push_to_hub=True,
    )
    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    trainer.save_model(training_args.output_dir)

    trainer.push_to_hub(dataset_name="HuggingFaceH4/Multilingual-Thinking")


def inference(**kwargs):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    gpu_prop = torch.cuda.get_device_properties("cuda")
    quantization_config = Mxfp4Config(dequantize=True)
    # Load the original model first
    model_kwargs = dict(
        torch_dtype="auto",
        use_cache=True,
        quantization_config=quantization_config,
        device_map="auto",
        attn_implementation="kernels-community/vllm-flash-attn3" if gpu_prop.major > 8 else "eager",
    )
    base_model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).cuda()

    # Merge fine-tuned weights with the base model
    peft_model_path = os.path.join(TRAIN_OUTPUT_DIR, "gpt-oss-20b-multilingual-reasoner")
    model = PeftModel.from_pretrained(base_model, peft_model_path)
    print(model)
    model = model.merge_and_unload()
    print(model)
    model.eval()

    with torch.inference_mode():
        REASONING_LANGUAGE = "German"
        SYSTEM_PROMPT = f"reasoning language: {REASONING_LANGUAGE}"
        USER_PROMPT = (
            "¿Cuál es el capital de Australia?"  # Spanish for "What is the capital of Australia?"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        gen_kwargs = {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": None,
            "top_k": None,
        }

        output_ids = model.generate(input_ids, **gen_kwargs)
        response = tokenizer.batch_decode(output_ids)[0]
        print(response)

        print("----" * 20)

        REASONING_LANGUAGE = "Chinese"  # or Hindi, or any other language...
        SYSTEM_PROMPT = f"reasoning language: {REASONING_LANGUAGE}"
        USER_PROMPT = "What is the national symbol of Canada?"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        output_ids = model.generate(input_ids, **gen_kwargs)
        response = tokenizer.batch_decode(output_ids)[0]
        print(response)


"""
IMAGE_GPU=H100 modal run src/train/trl/openai_gpt_oss.py --task train

IMAGE_GPU=L40s modal run src/train/trl/openai_gpt_oss.py --task inference
IMAGE_GPU=H100 modal run src/train/trl/openai_gpt_oss.py --task inference
"""


@app.local_entrypoint()
def main(
    task: str = "train",
):
    print(task)
    tasks = {
        "train": train,
        "inference": inference,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(tasks[task])
