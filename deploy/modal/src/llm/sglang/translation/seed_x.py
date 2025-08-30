import asyncio
import os
import subprocess
import modal

app = modal.App("seed-x_sglang")

img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvcr.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install(  # add sglang and some Python dependencies
        # as per sglang website: https://docs.sglang.ai/get_started/install.html
        "flashinfer-python",
        "sglang[all]>=0.5.1.post1",
        extra_options="--find-links https://flashinfer.ai/whl/cu126/torch2.6/flashinfer-python/",
        extra_index_url="https://flashinfer.ai/whl/cu126/torch2.6/",
    )
    .apt_install("libnuma-dev")  # Add NUMA library for sgl_kernel
    .env(
        {
            "TORCH_CUDA_ARCH_LIST": "7.5 8.0 8.6 8.7 8.9 9.0 10.0",
        }
    )
)

img = img.pip_install(
    "achatbot==0.0.23",
    extra_index_url="https://pypi.org/simple/",
)

HF_MODEL_DIR = "/root/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)


with img.imports():
    import uuid
    import torch
    from sglang import Engine
    from sglang.srt.managers.io_struct import GenerateReqInput

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
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
)
def run(func, **kwargs):
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    subprocess.run("which vllm", shell=True)
    if torch.cuda.is_available():
        gpu_prop = torch.cuda.get_device_properties("cuda")
        print(gpu_prop)

    func(**kwargs)


def generate(**kwargs):
    engine = Engine(
        model_path=model_path,
    )

    messages = [
        # without CoT
        "Translate the following English sentence into Chinese:\nMay the force be with you <zh>",
        # with CoT
        "Translate the following English sentence into Chinese and explain it in detail:\nMay the force be with you <zh>",
    ]
    for prompt in messages:
        iterator = engine.generate(  # no request_id param :(
            prompt=prompt,
            sampling_params={
                "n": 1,  # number of samples to generate
                "max_new_tokens": 512,
                "temperature": 0.95,
                "top_p": 1.0,
                "top_k": 20,
                "min_p": 0.0,
                # Penalizers
                "repetition_penalty": 1.1,
                "min_new_tokens": 0,
            },
            stream=True,
            return_logprob=True,
        )
        for part in iterator:
            print(part)
            meta_info = part["meta_info"]
            if "output_token_logprobs" in meta_info and len(meta_info["output_token_logprobs"]) > 0:
                token_id = meta_info["output_token_logprobs"][-1][1]
                print(token_id)


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    cpu=2.0,
    retries=0,
    image=img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
)
async def async_generate():
    engine = Engine(
        model_path=model_path,
    )

    messages = [
        # without CoT
        "Translate the following English sentence into Chinese:\nMay the force be with you <zh>",
        # with CoT
        "Translate the following English sentence into Chinese and explain it in detail:\nMay the force be with you <zh>",
    ]
    for prompt in messages:
        obj = GenerateReqInput(
            text=prompt,
            rid=str(uuid.uuid4().hex),
            sampling_params={
                "n": 1,  # number of samples to generate
                "max_new_tokens": 512,
                "temperature": 0.95,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.0,
                # Penalizers
                "repetition_penalty": 1.1,
                "min_new_tokens": 0,
            },
            stream=True,
            return_logprob=True,
        )
        generator = engine.tokenizer_manager.generate_request(obj, None)

        async for part in generator:
            print(part)
            print(part.get("output_ids", []))


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    cpu=2.0,
    retries=0,
    image=img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
)
async def run_achatbot_generator():
    import uuid
    import time

    from transformers import AutoTokenizer, GenerationConfig

    from achatbot.core.llm.sglang.generator import (
        SGlangGenerator,
        SGLangEngineArgs,
        ServerArgs,
    )
    from achatbot.common.session import Session
    from achatbot.common.types import SessionCtx

    generator = SGlangGenerator(
        **SGLangEngineArgs(
            serv_args=ServerArgs(model_path=model_path).__dict__,
        ).__dict__
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
    # session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)
    for case in prompt_cases:
        session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)
        session.ctx.state["token_ids"] = tokenizer.encode("hello, my name is")
        tokens = tokenizer(case["prompt"])
        session.ctx.state["token_ids"] = tokens["input_ids"]
        gen_kwargs = {**generation_config, **case["kwargs"], **tokens}
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


"""
IMAGE_GPU=L4 modal run src/llm/sglang/translation/seed_x.py
IMAGE_GPU=L4 modal run src/llm/sglang/translation/seed_x.py::async_generate
IMAGE_GPU=L4 modal run src/llm/sglang/translation/seed_x.py::run_achatbot_generator
"""


@app.local_entrypoint()
def main(
    task: str = "generate",
):
    print(task)
    tasks = {
        "generate": generate,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(
        tasks[task],
    )
