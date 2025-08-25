import datetime
import os
from pathlib import Path
import sys
import asyncio
import subprocess
import threading
import json
from time import perf_counter
import time


import modal


app = modal.App("seed-x_ctranslate2")
IMAGE_GPU = os.getenv("IMAGE_GPU", None)
APP_NAME = os.getenv("APP_NAME", "")

img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvcr.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "git-lfs")
    .pip_install(
        "ctranslate2",
        "transformers[torch]",
    )
    .env(
        {
            "LLM_MODEL": os.getenv("LLM_MODEL", "ByteDance-Seed/Seed-X-PPO-7B"),
        }
    )
)


HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)

with img.imports():
    import torch
    import ctranslate2
    import transformers

    MODEL_PATH = os.getenv("LLM_MODEL", "ByteDance-Seed/Seed-X-PPO-7B")
    model_path = os.path.join(HF_MODEL_DIR, MODEL_PATH)
    model_name = MODEL_PATH.split("/")[-1]
    ctranslate2_model_path = os.path.join(HF_MODEL_DIR, MODEL_PATH + "_ctranslate2")

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"


@app.function(
    gpu=IMAGE_GPU,
    cpu=2.0,
    retries=0,
    image=img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
    },
    secrets=[modal.Secret.from_name("achatbot")],
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
)
async def run(func, **kwargs):
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    subprocess.run("which vllm", shell=True)
    if torch.cuda.is_available():
        gpu_prop = torch.cuda.get_device_properties("cuda")
        print(gpu_prop)

    if asyncio.iscoroutinefunction(func):
        await func(**kwargs)
    else:
        func(**kwargs)


def transformers_convert(**kwargs):
    quantization = kwargs.get("quantization", "")
    cmd = f"ct2-transformers-converter --model {model_path} --output_dir {ctranslate2_model_path}"
    if quantization:
        # https://opennmt.net/CTranslate2/quantization.html
        cmd = f"ct2-transformers-converter --model {model_path} --quantization {quantization} --output_dir {ctranslate2_model_path}.{quantization}"
    print(cmd)
    subprocess.run(cmd, shell=True)


def tokenize(**kwargs):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    print(tokenizer)

    messages = [
        # without CoT
        "Translate the following English sentence into Chinese:\nMay the force be with you <zh>",
        # with CoT
        "Translate the following English sentence into Chinese and explain it in detail:\nMay the force be with you <zh>",
    ]
    token_ids = tokenizer.encode(messages)
    print(token_ids)

    tokens = tokenizer.decode(token_ids)
    print(tokens)

    start_tokens = tokenizer.convert_ids_to_tokens(token_ids)
    print(start_tokens)


def generate(**kwargs):
    print(f"{device=}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    quantization = kwargs.get("quantization", "")
    model_dir = ctranslate2_model_path
    if quantization:
        # https://opennmt.net/CTranslate2/quantization.html
        model_dir = f"{ctranslate2_model_path}.{quantization}"
    generator = ctranslate2.Generator(model_dir, device=device)

    messages = [
        # without CoT
        "Translate the following English sentence into Chinese:\nMay the force be with you <zh>",
        # with CoT
        "Translate the following English sentence into Chinese and explain it in detail:\nMay the force be with you <zh>",
    ]
    for message in messages:
        token_ids = tokenizer.encode(message)
        print(token_ids)

        start_tokens = tokenizer.convert_ids_to_tokens(token_ids)
        print(start_tokens)

        # https://opennmt.net/CTranslate2/generation.html
        # https://opennmt.net/CTranslate2/python/ctranslate2.Generator.html#ctranslate2.Generator.generate_tokens
        step_results = generator.generate_tokens(
            start_tokens,
            end_token=tokenizer.eos_token,
            sampling_temperature=0.6,
            sampling_topk=20,
            sampling_topp=1,
        )

        first = True
        start_time = time.perf_counter()
        for step_result in step_results:
            if first:
                ttft = time.perf_counter() - start_time
                print(f"generate TTFT time: {ttft} s")
                first = False
            # print(step_result)
            if step_result.token_id == tokenizer.bos_token_id:
                continue
            if step_result.token_id == tokenizer.eos_token_id:
                break
            print(step_result.token, end="", flush=True)


def generate_batch(**kwargs):
    print(f"{device=}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    generator = ctranslate2.Generator(ctranslate2_model_path, device=device)

    messages = [
        # without CoT
        "Translate the following English sentence into Chinese:\nMay the force be with you <zh>",
        # with CoT
        # "Translate the following English sentence into Chinese and explain it in detail:\nMay the force be with you <zh>",
    ]

    def generation_callback(step_result: ctranslate2.GenerationStepResult) -> bool:
        print(step_result)
        return False

    for message in messages:
        token_ids = tokenizer.encode(message)
        print(token_ids)

        start_tokens = tokenizer.convert_ids_to_tokens(token_ids)
        print(start_tokens)

        # use generate_batch to beam search, and callback to iter step
        # https://opennmt.net/CTranslate2/python/ctranslate2.Generator.html#ctranslate2.Generator.generate_batch
        res = generator.generate_batch(
            [start_tokens],
            max_batch_size=2,
            include_prompt_in_result=False,
            end_token=tokenizer.eos_token,
            sampling_temperature=0.6,
            sampling_topk=20,
            sampling_topp=1,
            # beam_search
            beam_size=2,
            # callback=generation_callback, # beam_size=1 for greedy decoding, need callback
        )
        print(res)


"""
modal run src/llm/ctranslate2/seed_x.py --task tokenize
modal run src/llm/ctranslate2/seed_x.py --task transformers_convert
modal run src/llm/ctranslate2/seed_x.py --task transformers_convert --quantization int8
modal run src/llm/ctranslate2/seed_x.py --task transformers_convert --quantization int8_bfloat16

# CPU 
modal run src/llm/ctranslate2/seed_x.py --task generate
modal run src/llm/ctranslate2/seed_x.py --task generate --quantization int8
modal run src/llm/ctranslate2/seed_x.py --task generate --quantization int8_bfloat16

# L4 GPU
IMAGE_GPU=L4 modal run src/llm/ctranslate2/seed_x.py --task generate
IMAGE_GPU=L4 modal run src/llm/ctranslate2/seed_x.py --task generate --quantization int8
IMAGE_GPU=L4 modal run src/llm/ctranslate2/seed_x.py --task generate --quantization int8_bfloat16

# offline batch with beam search
IMAGE_GPU=L4 modal run src/llm/ctranslate2/seed_x.py --task generate_batch
"""


@app.local_entrypoint()
def main(
    task: str = "generate",
    quantization: str = "",
):
    print(task)
    tasks = {
        "tokenize": tokenize,
        "transformers_convert": transformers_convert,
        "generate": generate,
        "generate_batch": generate_batch,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(
        tasks[task],
        quantization=quantization,
    )
