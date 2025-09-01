import os
import sys
import json
import uuid
import time
import asyncio
import datetime
import threading
import subprocess
from pathlib import Path
from time import perf_counter

import modal


app = modal.App("seed-x_trtllm")
IMAGE_GPU = os.getenv("IMAGE_GPU", None)
BACKEND = os.getenv("BACKEND", "")
APP_NAME = os.getenv("APP_NAME", "")
TP = os.getenv("TP", "1")

GIT_TAG_OR_HASH = "0.18.0"

img = (
    # https://nvidia.github.io/TensorRT-LLM/release-notes.html
    # https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release/tags
    # https://modal.com/docs/examples/trtllm_latency
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",  # TRT-LLM requires Python 3.12
    )
    .entrypoint([])  # remove verbose logging by base image on entry
    .apt_install("openmpi-bin", "libopenmpi-dev", "git", "git-lfs", "wget")
    .pip_install(
        f"tensorrt-llm=={GIT_TAG_OR_HASH}",
        "pynvml<12",  # avoid breaking change to pynvml version API
        "flashinfer-python==0.2.5",
        "cuda-python==12.9.1",
        pre=True,
        extra_index_url="https://pypi.nvidia.com",
    )
    .env({"TORCH_CUDA_ARCH_LIST": "8.0 8.9 9.0 9.0a"})
)

img = img.pip_install(
    "achatbot==0.0.24.post1",
    extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
).env(
    {
        "TLLM_LLMAPI_BUILD_CACHE": "1",
    }
)


HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)

TRT_MODEL_DIR = "/root/.achatbot/trt_models"
trt_model_vol = modal.Volume.from_name("triton_trtllm_models", create_if_missing=True)

TRT_MODEL_CACHE_DIR = "/tmp/.cache/tensorrt_llm/llmapi/"
trt_model_cache_vol = modal.Volume.from_name("triton_trtllm_cache_models", create_if_missing=True)


with img.imports():
    import torch

    MODEL_PATH = os.getenv("LLM_MODEL", "ByteDance-Seed/Seed-X-PPO-7B")
    model_path = os.path.join(HF_MODEL_DIR, MODEL_PATH)
    model_name = MODEL_PATH.split("/")[-1]


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    cpu=2.0,
    retries=0,
    image=img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        TRT_MODEL_CACHE_DIR: trt_model_cache_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
)
async def run_achatbot_generator():
    from transformers import AutoTokenizer, GenerationConfig
    from tensorrt_llm.llmapi import KvCacheConfig

    from achatbot.core.llm.tensorrt_llm.generator import (
        TrtLLMGenerator,
        TensorRTLLMEngineArgs,
        LMGenerateArgs,
        LlmArgs,
    )
    from achatbot.common.types import SessionCtx
    from achatbot.common.session import Session
    from achatbot.common.logger import Logger

    Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)

    generator = TrtLLMGenerator(
        **TensorRTLLMEngineArgs(
            serv_args=LlmArgs(
                model=model_path,
                # kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.7),
                kv_cache_config={"free_gpu_memory_fraction": 0.7},
            ).to_dict(),
            gen_args=LMGenerateArgs(lm_gen_stops=None).__dict__,
        ).__dict__,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    generation_config = {}
    if os.path.exists(os.path.join(model_path, "generation_config.json")):
        generation_config = GenerationConfig.from_pretrained(
            model_path, "generation_config.json"
        ).to_dict()

    prompt_cases = [
        # without CoT
        {
            "prompt": "Translate the following English sentence into Chinese:\nMay the force be with you <zh>",
            "kwargs": {"max_new_tokens": 30, "stop_ids": [2]},
        },
        # with CoT
        {
            "prompt": "Translate the following English sentence into Chinese and explain it in detail:\nMay the force be with you <zh>",
            "kwargs": {"max_new_tokens": 512, "stop_ids": [2]},
        },
    ]

    # test the same session
    # session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)
    for case in prompt_cases:
        session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)
        tokens = tokenizer(case["prompt"])
        session.ctx.state["token_ids"] = tokens.pop("input_ids")
        # gen_kwargs = {**generation_config, **case["kwargs"], **tokens}
        gen_kwargs = case["kwargs"]
        print("gen_kwargs:", gen_kwargs)
        first = True
        start_time = time.perf_counter()
        gen_texts = ""
        async for token_id in generator.generate(session, **gen_kwargs):
            if first:
                ttft = time.perf_counter() - start_time
                print(f"generate TTFT time: {ttft} s")
                first = False
            gen_text = tokenizer.decode(token_id)
            gen_texts += gen_text
            print(session.ctx.client_id, token_id, gen_text)
        print(session.ctx.client_id, gen_texts)

    generator.close()


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    cpu=2.0,
    retries=0,
    image=img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        TRT_MODEL_DIR: trt_model_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
)
async def run_achatbot_runner_generator(
    app_name: str = "seed-x",
    trt_dtype: str = "bfloat16",
):
    from transformers import AutoTokenizer, GenerationConfig

    from achatbot.core.llm.tensorrt_llm.generator import (
        TrtLLMRunnerGenerator,
        TensorRTLLMRunnerArgs,
        TensorRTLLMRunnerEngineArgs,
        LMGenerateArgs,
    )
    from achatbot.common.types import SessionCtx
    from achatbot.common.session import Session
    from achatbot.common.logger import Logger

    Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)

    engine_dir = os.path.join(TRT_MODEL_DIR, app_name, f"trt_engines_{trt_dtype}")
    generator = TrtLLMRunnerGenerator(
        **TensorRTLLMRunnerEngineArgs(
            serv_args=TensorRTLLMRunnerArgs(engine_dir=engine_dir).__dict__
        ).__dict__,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    generation_config = {}
    if os.path.exists(os.path.join(model_path, "generation_config.json")):
        generation_config = GenerationConfig.from_pretrained(
            model_path, "generation_config.json"
        ).to_dict()

    prompt_cases = [
        # without CoT
        {
            "prompt": "Translate the following English sentence into Chinese:\nMay the force be with you <zh>",
            "kwargs": {"max_new_tokens": 30, "stop_ids": [2]},
        },
        # with CoT
        {
            "prompt": "Translate the following English sentence into Chinese and explain it in detail:\nMay the force be with you <zh>",
            "kwargs": {"max_new_tokens": 512, "stop_ids": [2]},
        },
    ]

    # test the same session
    # session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)
    for case in prompt_cases:
        session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)
        tokens = tokenizer(case["prompt"])
        session.ctx.state["token_ids"] = tokens["input_ids"]
        # gen_kwargs = {**generation_config, **case["kwargs"], **tokens}
        gen_kwargs = {**case["kwargs"], **tokens}
        print("gen_kwargs:", gen_kwargs)
        first = True
        start_time = time.perf_counter()
        gen_texts = ""
        async for token_id in generator.generate(session, **gen_kwargs):
            if first:
                ttft = time.perf_counter() - start_time
                print(f"generate TTFT time: {ttft} s")
                first = False
            gen_text = tokenizer.decode(token_id)
            gen_texts += gen_text
            print(session.ctx.client_id, token_id, gen_text)
        print(session.ctx.client_id, gen_texts)

    generator.close()


"""
# covert -> build(BuildConfig) -> load -> run
IMAGE_GPU=L4 modal run src/llm/trtllm/translation/seed_x.py::run_achatbot_generator

# https://github.com/NVIDIA/TensorRT-LLM/blob/v0.18.0/examples/mixtral/README.md
# covert to trtllm ckpt; and build engine
modal run src/llm/trtllm/translation/compile_model.py \
    --app-name "seed-x" \
    --hf-repo-dir "ByteDance-Seed/Seed-X-PPO-7B" \
    --trt-dtype "bfloat16" \
    --convert-other-args "" \
    --compile-other-args "--max_batch_size 16 --max_num_tokens 32768"

# engine runner
IMAGE_GPU=L4 modal run src/llm/trtllm/translation/seed_x.py::run_achatbot_runner_generator \
    --app-name "seed-x" \
    --trt-dtype "bfloat16"


"""
