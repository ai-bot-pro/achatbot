import os
import io
import uuid
import time
import asyncio
import requests
import subprocess
from time import perf_counter
from threading import Thread


import modal

APP_NAME = os.getenv("APP_NAME", "")

app = modal.App("seed_x")
img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "clang", "cmake", "ninja-build")
    .pip_install("wheel")
    .pip_install("accelerate", "torch==2.6.0", "transformers==4.56.0")
    # https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1
    .pip_install("flash-attn==2.7.4.post1", extra_options="--no-build-isolation")
    .pip_install("compressed-tensors==0.11.0")
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TQDM_DISABLE": "1",
            "LLM_MODEL": os.getenv("LLM_MODEL", "ByteDance-Seed/Seed-X-PPO-7B"),
        }
    )
)

if APP_NAME == "achatbot":
    img = img.pip_install(
        f"achatbot==0.0.24.post2",
        extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
    )

HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)


with img.imports():
    import random

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.generation.streamers import TextIteratorStreamer

    model_name = os.getenv("LLM_MODEL")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)


@app.function(
    gpu=os.getenv("IMAGE_GPU", None),
    cpu=2.0,
    retries=1,
    image=img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
)
async def run(func):
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    gpu_prop = None
    if torch.cuda.is_available():
        gpu_prop = torch.cuda.get_device_properties("cuda")
        print(gpu_prop)

    if asyncio.iscoroutinefunction(func):
        await func(gpu_prop)
    else:
        func(gpu_prop)


def print_model_params(model: torch.nn.Module, extra_info=""):
    # print the number of parameters in the model
    model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(model)
    print(f"{extra_info} {model_million_params} M parameters")


# https://huggingface.co/collections/ByteDance-Seed/seed-x-6878753f2858bc17afa78543
def dump_model(gpu_prop):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
    )

    model = model.eval()
    print_model_params(model, f"{model_name}")
    print(f"{model.config=}")

    del model
    torch.cuda.empty_cache()


def tokenize(gpu_prop):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True, legacy=False)
    print(f"{tokenizer=}")
    tokenizer.add_special_tokens({"pad_token": "</s>"})
    print(f"add pad {tokenizer.pad_token_id=} {tokenizer=}")

    messages = [
        # without CoT
        "Translate the following English sentence into Chinese:\nMay the force be with you <zh>",
        # with CoT
        "Translate the following English sentence into Chinese and explain it in detail:\nMay the force be with you <zh>",
    ]

    print(messages)

    inputs = tokenizer(
        messages,
        return_tensors="pt",
        padding=True,  # for batch
        # truncation=True,  # for batch
    )
    for key, value in inputs.items():
        print(f"{key}: {value.shape=}")
    input_ids = inputs["input_ids"]
    prompt = tokenizer.batch_decode(input_ids)
    print(f"{prompt=}")


def generate(gpu_prop):
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True, legacy=False)
    tokenizer.add_special_tokens({"pad_token": "</s>"})
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
    )
    model = model.eval()
    # print(f"{model.config=}")

    messages = [
        # without CoT
        "Translate the following English sentence into Chinese:\nMay the force be with you <zh>",
        "Translate the following Chinese sentence into English:\n奥利给 <zh>",
        # with CoT
        "Translate the following English sentence into Chinese and explain it in detail:\nMay the force be with you <zh>",
    ]

    print(messages)

    inputs = tokenizer(
        messages,
        return_tensors="pt",
        padding=True,  # for batch
        # truncation=True,  # for batch
    ).to(model.device)
    for key, value in inputs.items():
        print(f"{key}: {value.shape=}")
    input_ids = inputs["input_ids"]
    prompt = tokenizer.batch_decode(input_ids)
    print(f"{prompt=}")
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        # https://huggingface.co/docs/transformers/generation_strategies
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            num_beams=4,  # beam search generate slow, with sampling
            do_sample=True,
            temperature=0.7,
            top_k=20,
            top_p=0.6,
            repetition_penalty=1.05,
            max_new_tokens=64,
            pad_token_id=tokenizer.pad_token_id,  # for batch
        )
        generated_ids = [generated_id[input_len:] for generated_id in generated_ids]

    for generated_id in generated_ids:
        print(f"{generated_id.shape=}")
    generated_text = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    print(generated_text)


def generate_stream(gpu_prop):
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True, legacy=False)
    tokenizer.add_special_tokens({"pad_token": "</s>"})
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
    )
    model = model.eval()
    # print(f"{model.config=}")

    inputs = tokenizer(
        "Translate the following English sentence into Chinese:\nMay the force be with you <zh>",
        # "Translate the following English sentence into Chinese and explain it in detail:\nMay the force be with you <zh>",
        return_tensors="pt",
        padding=True,  # for batch
        # truncation=True,  # for batch
    ).to(model.device)
    for key, value in inputs.items():
        print(f"{key}: {value.shape=}")
    input_ids = inputs["input_ids"]
    prompt = tokenizer.batch_decode(input_ids)
    print(f"{prompt=}")

    streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        input_ids=input_ids,
        attention_mask=inputs["attention_mask"],
        # num_beams=4,  # beam search generate slow, with sampling, TextStreamer only supports batch size 1
        do_sample=True,
        temperature=0.7,
        top_k=20,
        top_p=0.6,
        repetition_penalty=1.05,
        max_new_tokens=64,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    generated_text = ""
    start = perf_counter()
    times = []
    with torch.inference_mode():
        for new_text in streamer:
            times.append(perf_counter() - start)
            print(new_text, end="", flush=True)
            generated_text += new_text
            start = perf_counter()
    print(f"\n{generated_text=} TTFT: {times[0]:.2f}s total time: {sum(times):.2f}s")


async def achatbot_gen_stream(gpu_prop):
    from achatbot.core.llm.transformers.generator import TransformersGenerator, GenerationConfig
    from achatbot.types.llm.transformers import TransformersLMArgs
    from achatbot.common.types import MODELS_DIR, SessionCtx
    from achatbot.common.session import Session
    from achatbot.common.logger import Logger

    Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True, legacy=False)
    tokens = tokenizer(
        "Translate the following English sentence into Chinese:\nMay the force be with you <zh>"
    )
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    # generator
    generator = TransformersGenerator(
        **TransformersLMArgs(lm_model_name_or_path=MODEL_PATH).__dict__
    )

    # generation_config
    generation_config = GenerationConfig.from_pretrained(MODEL_PATH, "generation_config.json")
    generation_config.max_new_tokens = 64
    # generation_config.num_beams = 4 #for batch, TextStreamer only supports batch size 1
    print(generation_config.to_dict())

    session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)
    session.ctx.state["token_ids"] = input_ids
    first = True
    start_time = time.perf_counter()
    async for token_id in generator.generate(
        session,
        attention_mask=attention_mask,
        stop_ids=[tokenizer.eos_token_id],
        **generation_config.to_dict(),
    ):
        if first:
            ttft = time.perf_counter() - start_time
            print(f"generate TTFT time: {ttft} s")
            first = False
        gen_text = tokenizer.decode(token_id)
        print(token_id, gen_text)


"""
modal run src/llm/transformers/translation/seed_x.py --task tokenize

IMAGE_GPU=T4 modal run src/llm/transformers/translation/seed_x.py --task dump_model

IMAGE_GPU=T4 modal run src/llm/transformers/translation/seed_x.py --task generate
IMAGE_GPU=L4 modal run src/llm/transformers/translation/seed_x.py --task generate

IMAGE_GPU=T4 modal run src/llm/transformers/translation/seed_x.py --task generate_stream
IMAGE_GPU=L4 modal run src/llm/transformers/translation/seed_x.py --task generate_stream

APP_NAME=achatbot IMAGE_GPU=T4 modal run src/llm/transformers/translation/seed_x.py --task achatbot_gen_stream
APP_NAME=achatbot IMAGE_GPU=L4 modal run src/llm/transformers/translation/seed_x.py --task achatbot_gen_stream
"""


@app.local_entrypoint()
def main(task: str = "dump_model"):
    tasks = {
        "dump_model": dump_model,
        "tokenize": tokenize,
        "generate": generate,
        "generate_stream": generate_stream,
        "achatbot_gen_stream": achatbot_gen_stream,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(tasks[task])
