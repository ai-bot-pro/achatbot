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


app = modal.App("hunyuan-mt_trtllm")
IMAGE_GPU = os.getenv("IMAGE_GPU", None)
BACKEND = os.getenv("BACKEND", "")
APP_NAME = os.getenv("APP_NAME", "")
TP = os.getenv("TP", "1")

# NOTE: version must >=0.19.0
GIT_TAG_OR_HASH = "1.0.0rc6"

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
        # "pynvml<12",  # avoid breaking change to pynvml version API
        "flashinfer-python==0.2.5",
        "cuda-python==12.9.1",
        pre=True,
        extra_index_url="https://pypi.nvidia.com",
    )
    .env({"TORCH_CUDA_ARCH_LIST": "8.0 8.9 9.0 9.0a"})
)

img = (
    img.pip_install(
        "achatbot==0.0.24.post2",
        extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
    )
    # https://github.com/NVIDIA/TensorRT-Model-Optimizer/issues/280
    .pip_install("transformers==4.56.0", "nvidia-modelopt[torch]~=0.35.0")
    .env(
        {
            "TLLM_LLMAPI_BUILD_CACHE": "1",
        }
    )
)


HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)

TRT_MODEL_DIR = "/root/.achatbot/trt_models"
trt_model_vol = modal.Volume.from_name("triton_trtllm_models", create_if_missing=True)

TRT_MODEL_CACHE_DIR = "/tmp/.cache/tensorrt_llm/llmapi/"
trt_model_cache_vol = modal.Volume.from_name("triton_trtllm_cache_models", create_if_missing=True)


with img.imports():
    import torch

    # sys.path.insert(0, "/TensorRT-LLM")
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs, TrtLlmArgs, LoadFormat
    from tensorrt_llm.bindings.executor import KvCacheConfig

    MODEL_ID = os.getenv("LLM_MODEL", "tencent/Hunyuan-MT-7B")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, MODEL_ID)
    model_name = MODEL_ID.split("/")[-1]


@app.function(
    gpu=IMAGE_GPU,
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


def cover_load_format(v):
    load_format = "AUTO"
    if isinstance(v, int):
        return LoadFormat(v)
    elif isinstance(v, str):
        load_format = v.upper()
        if load_format not in LoadFormat.__members__:
            raise ValueError(f"Invalid LoadFormat: {v}")
    return LoadFormat(load_format)


def generate(**kwargs):
    """
    https://nvidia.github.io/TensorRT-LLM/_modules/tensorrt_llm/llmapi/llm.html#LLM.generate
    """

    prompts = [
        f"<|startoftext|>把下面的文本翻译成English，不要额外解释。 \n\n你好<|extra_0|>",
    ]

    # load hf model, flashinfer.jit compile
    # https://nvidia.github.io/TensorRT-LLM/1.0.0rc1/
    llm = LLM(
        model=MODEL_PATH,
        max_batch_size=2,
        max_seq_len=512,
        kv_cache_config={"free_gpu_memory_fraction": 0.5},
    )

    # # https://github.com/NVIDIA/TensorRT-LLM/blob/v1.0.0rc1/tensorrt_llm/llmapi/llm_args.py
    # #  TrtLlmArgs Value error, Inferred model format _ModelFormatKind.HF, but failed to load config.json: The given huggingface model architecture HunYuanDenseV1ForCausalLM is not supported in TRT-LLM yet [type=value_error, input_value={'model': '/root/.achatbo.../tencent/Hunyuan-MT-7B'}, input_type=dict]
    # # serv_args = TrtLlmArgs.from_kwargs(**{"model": MODEL_PATH})
    # serv_args = TorchLlmArgs.from_kwargs(
    #     **{
    #         "model": MODEL_PATH,
    #         # "load_format": "auto",
    #     }
    # )
    # print(serv_args, serv_args.to_dict())
    # args = serv_args.to_dict()
    # args["load_format"] = cover_load_format(args["load_format"])
    # print(args)
    # llm = LLM(**args)

    # https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.SamplingParams
    sampling_params = SamplingParams(
        temperature=0.7, top_k=20, top_p=0.6, max_tokens=256, repetition_penalty=1.05
    )

    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    # Print the outputs.
    for output in outputs:
        print(output)

    llm.shutdown()


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

    args = {
        "serv_args": {
            "model": MODEL_PATH,
            "kv_cache_config": {
                "free_gpu_memory_fraction": 0.7,
            },
        },
        "gen_args": {"lm_gen_stops": None},
    }
    generator = TrtLLMGenerator(**args)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    generation_config = {}
    if os.path.exists(os.path.join(MODEL_PATH, "generation_config.json")):
        generation_config = GenerationConfig.from_pretrained(
            MODEL_PATH, "generation_config.json"
        ).to_dict()

    prompt_cases = [
        {
            "prompt": "<|startoftext|>把下面的文本翻译成English，不要额外解释。\n\n9月3日，看了阅兵仪式，东风快递好牛叉！<|extra_0|>",
            "kwargs": {"max_new_tokens": 64, "stop_ids": [127960]},
        },
        # {
        #    "prompt": """<|startoftext|>把下面的文本翻译成English，不要额外解释。\n\n9月3日上午九点，中国的抗日战争胜利纪念日暨世界反法西斯战争胜利80周年纪念活动正式登场。\n 升旗仪式后，中国领导人习近平发表讲话称，“全军将士要忠实履行神圣职责，加快建设世界一流军队，坚决维护国家主权统一、领土完整。为实现中华民族伟大复兴提供战略支撑，为世界和平与发展作出更大贡献。”\n 相比十年之前的“9·3 ”阅兵，习近平此次的讲话并未透露更多信息，当时他在讲话中强调，中国永不称霸，并宣布将裁军30万。\n 十年间，中国经历了被认为是中共建政以来最大力度的军队体制改革，服役人数稳定在 200万人，从七大军区改为五大战区，大量将领被整肃，军费则上涨超过70%。\n 此次纪念大会和阅兵式是十年来第二次“九三阅兵”，也是习近平上任后第三次天安门阅兵。阅兵式上，展示的新式武器更多，尤其是能搭载核弹头的战略导弹，以及大量无人作战武器。\n<|extra_0|>""",
        #    "kwargs": {"max_new_tokens": 1024, "stop_ids": [127960]},
        # },
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


"""
# https://github.com/Tencent-Hunyuan/Hunyuan-7B#tensorrt-llm

IMAGE_GPU=L4 modal run src/llm/trtllm/translation/hunyuan_mt.py --task generate
IMAGE_GPU=L4 modal run src/llm/trtllm/translation/hunyuan_mt.py --task run_achatbot_generator
"""


@app.local_entrypoint()
def main(
    task: str = "generate",
):
    print(task)
    tasks = {
        "generate": generate,
        "run_achatbot_generator": run_achatbot_generator,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(tasks[task])
