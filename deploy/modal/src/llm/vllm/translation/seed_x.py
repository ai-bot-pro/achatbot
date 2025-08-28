import datetime
import os
from pathlib import Path
import sys
import asyncio
import subprocess
import threading
import json
from time import perf_counter


import modal


app = modal.App("seed-x_vllm")
IMAGE_GPU = os.getenv("IMAGE_GPU", None)
BACKEND = os.getenv("BACKEND", "")
APP_NAME = os.getenv("APP_NAME", "")
TP = os.getenv("TP", "1")

img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvcr.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "git-lfs")
    .pip_install(
        "vllm==0.8.0",
        # extra_index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install("transformers==4.51.3")
)
# NOTE: use Beam search or Greedy decoding, don't use flashinfer for top-p & top-k sampling.
if BACKEND == "flashinfer":
    img = img.pip_install(
        f"flashinfer-python",
        # extra_index_url="https://flashinfer.ai/whl/cu126/torch2.6/",
    )
img = img.env(
    {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # "TQDM_DISABLE": "1",
        "LLM_MODEL": os.getenv("LLM_MODEL", "ByteDance-Seed/Seed-X-PPO-7B"),
        "VLLM_USE_V1": os.getenv("VLLM_USE_V1", "1"),
        "TP": TP,
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "TORCH_CUDA_ARCH_LIST": "8.0 8.9",
        # "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
    }
)
img = img.pip_install(
    "achatbot==0.0.24",
    extra_index_url="https://pypi.org/simple/",
)

HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
VLLM_CACHE_DIR = "/root/.cache/vllm"
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

with img.imports():
    import torch
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import BeamSearchParams

    MODEL_PATH = os.getenv("LLM_MODEL", "ByteDance-Seed/Seed-X-PPO-7B")
    model_path = os.path.join(HF_MODEL_DIR, MODEL_PATH)
    model_name = MODEL_PATH.split("/")[-1]


@app.function(
    gpu=IMAGE_GPU,
    cpu=2.0,
    retries=0,
    image=img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        VLLM_CACHE_DIR: vllm_cache_vol,
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


def generate(**kwargs):
    model = LLM(
        model=model_path,
        max_num_seqs=512,
        tensor_parallel_size=int(os.getenv("TP", "1")),
        enable_prefix_caching=True,
        gpu_memory_utilization=0.95,
        task="generate",
    )

    messages = [
        # without CoT
        "Translate the following English sentence into Chinese:\nMay the force be with you <zh>",
        # with CoT
        "Translate the following English sentence into Chinese and explain it in detail:\nMay the force be with you <zh>",
    ]
    # Greedy decoding
    decoding_params = SamplingParams(temperature=0, max_tokens=512, skip_special_tokens=True)
    # Random sampling
    decoding_params = SamplingParams(
        temperature=0.6, top_k=20, top_p=1.0, max_tokens=512, skip_special_tokens=True
    )
    results = model.generate(messages, decoding_params)

    responses = [res.outputs[0].text.strip() for res in results]

    print(responses)


def beam_search(**kwargs):
    model = LLM(
        model=model_path,
        max_num_seqs=512,
        tensor_parallel_size=int(os.getenv("TP", "1")),
        # enable_prefix_caching=True,
        # disable_sliding_window=True,
        gpu_memory_utilization=0.95,
        max_model_len=28656,
        task="generate",
    )

    messages = [
        # without CoT
        {
            "prompt": "Translate the following English sentence into Chinese:\nMay the force be with you <zh>",
        },
        # with CoT
        # {
        #    "prompt": "Translate the following English sentence into Chinese and explain it in detail:\nMay the force be with you <zh>",
        # },
    ]

    # Beam search (We recommend using beam search decoding) ,
    # V1 does not yet support Pooling models.
    # don't need VLLM_USE_V1 to set

    # so ugly docs, look at the source code :)
    # https://github.com/vllm-project/vllm/blob/v0.8.0/vllm/entrypoints/llm.py#L505
    # generate 2 * beam_width candidates at each step
    # following the huggingface transformers implementation
    # at https://github.com/huggingface/transformers/blob/e15687fffe5c9d20598a19aeab721ae0a7580f8a/src/transformers/generation/beam_search.py#L534 # noqa
    decoding_params = BeamSearchParams(beam_width=2, max_tokens=512)  # rank = beam_width*2
    results = model.beam_search(messages, decoding_params)
    # BeamSearchSequence: https://github.com/vllm-project/vllm/blob/v0.8.0/vllm/beam_search.py#L13
    for output in results:
        text = [sequence.text for sequence in output.sequences]
        print(text)


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    cpu=2.0,
    retries=0,
    image=img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        VLLM_CACHE_DIR: vllm_cache_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=100,
)
def run_sync():
    engine = LLM(
        model=model_path,
        max_num_seqs=512,
        tensor_parallel_size=int(os.getenv("TP", "1")),
        enable_prefix_caching=True,
        gpu_memory_utilization=0.95,
        task="generate",
    )

    sampling_params = engine.get_default_sampling_params()
    print(sampling_params)
    sampling_params.n = 1
    sampling_params.seed = 42
    sampling_params.max_model_len = 10240
    sampling_params.max_tokens = 512
    sampling_params.temperature = 0.9
    sampling_params.top_p = 1.0
    sampling_params.top_k = 50
    sampling_params.min_p = 0.0
    # Penalizers
    sampling_params.repetition_penalty = 1.1
    sampling_params.min_tokens = 0

    messages = [
        # without CoT
        "Translate the following English sentence into Chinese:\nMay the force be with you <zh>",
        # with CoT
        "Translate the following English sentence into Chinese and explain it in detail:\nMay the force be with you <zh>",
    ]
    for prompt in messages:
        outputs = engine.generate(
            prompt,
            sampling_params=sampling_params,
        )
        print(outputs)
        i = 1
        for part in outputs:
            print(part)
            if i == 3:
                print("stopping")
                break
            i += 1


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    cpu=2.0,
    retries=0,
    image=img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        VLLM_CACHE_DIR: vllm_cache_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
)
async def run_async():
    import asyncio
    import uuid

    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    # from vllm.engine.async_llm_engine import AsyncEngineArgs, AsyncLLMEngine

    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model=model_path,
            max_num_seqs=512,
            tensor_parallel_size=int(os.getenv("TP", "1")),
            enable_prefix_caching=True,
            gpu_memory_utilization=0.95,
            task="generate",
        )
    )
    sampling_params = SamplingParams(
        n=1,
        seed=42,
        max_tokens=512,
        temperature=0.9,
        top_p=1.0,
        top_k=50,
        min_p=0.0,
        # Penalizers
        repetition_penalty=1.1,
        min_tokens=0,
    )
    print(sampling_params)

    messages = [
        # without CoT
        "Translate the following English sentence into Chinese:\nMay the force be with you <zh>",
        # with CoT
        "Translate the following English sentence into Chinese and explain it in detail:\nMay the force be with you <zh>",
    ]
    for prompt in messages:
        outputs_generator = engine.generate(
            prompt, sampling_params=sampling_params, request_id=str(uuid.uuid4().hex)
        )
        async for part in outputs_generator:
            print(part)
            """
            RequestOutput(request_id=a80d0cce8c774cbaada1ae0bb993ddf5, prompt='Hello, my name is', prompt_token_ids=[9707, 11, 847, 829, 374], encoder_prompt=None, encoder_prompt_token_ids=None, prompt_logprobs=None, outputs=[CompletionOutput(index=0, text=' David', token_ids=[6798], cumulative_logprob=None, logprobs=None, finish_reason=None, stop_reason=None)], finished=False, metrics=None, lora_request=None, num_cached_tokens=None, multi_modal_placeholders={})
            """


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    cpu=2.0,
    retries=0,
    image=img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        VLLM_CACHE_DIR: vllm_cache_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=100,
)
async def run_achatbot_generator():
    import uuid
    import os
    import time

    from achatbot.core.llm.vllm.generator import VllmGenerator, VllmEngineArgs, AsyncEngineArgs
    from achatbot.common.types import SessionCtx
    from achatbot.common.session import Session
    from transformers import AutoTokenizer, GenerationConfig

    generator = VllmGenerator(
        **VllmEngineArgs(
            serv_args=AsyncEngineArgs(
                model=model_path,
                max_num_seqs=512,
                tensor_parallel_size=int(os.getenv("TP", "1")),
                enable_prefix_caching=True,
                gpu_memory_utilization=0.92,
                task="generate",
            ).__dict__
        ).__dict__,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    generation_config = {}
    if os.path.exists(os.path.join(model_path, "generation_config.json")):
        generation_config = GenerationConfig.from_pretrained(
            model_path, "generation_config.json"
        ).to_dict()
    # async def run():
    prompt_cases = [
        # without CoT
        {
            "prompt": "Translate the following English sentence into Chinese:\nMay the force be with you <zh>",
            "kwargs": {"max_new_tokens": 30, "stop_ids": []},
        },
        # with CoT
        {
            "prompt": "Translate the following English sentence into Chinese and explain it in detail:\nMay the force be with you <zh>",
            "kwargs": {"max_new_tokens": 512, "stop_ids": []},
        },
    ]
    # test the same session
    session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)
    for case in prompt_cases:
        # session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)
        tokens = tokenizer(case["prompt"])
        session.ctx.state["token_ids"] = tokens["input_ids"]
        # gen_kwargs = {**generation_config, **case["kwargs"], **tokens} # hack test, vllm have some bug :)
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
            # print(session.ctx.client_id, token_id, gen_text)
        print(session.ctx.client_id, gen_texts)


"""
IMAGE_GPU=L4 modal run src/llm/vllm/translation/seed_x.py
IMAGE_GPU=L4 VLLM_USE_V1=0 modal run src/llm/vllm/translation/seed_x.py --task beam_search

#IMAGE_GPU=L4 BACKEND=flashinfer modal run src/llm/vllm/translation/seed_x.py


IMAGE_GPU=L4 modal run src/llm/vllm/translation/seed_x.py::run_sync
IMAGE_GPU=L4 modal run src/llm/vllm/translation/seed_x.py::run_async
IMAGE_GPU=L4 modal run src/llm/vllm/translation/seed_x.py::run_achatbot_generator
"""


@app.local_entrypoint()
def main(
    task: str = "generate",
):
    print(task)
    tasks = {
        "generate": generate,
        "beam_search": beam_search,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(
        tasks[task],
    )
