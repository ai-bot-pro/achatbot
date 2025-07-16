import asyncio
import os
import subprocess
from typing import Optional
import uuid
import time
import requests
import io

import modal

BACKEND = os.getenv("BACKEND", "")
APP_NAME = os.getenv("APP_NAME", "")
TP = os.getenv("TP", "1")

vllm_image = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "cmake", "ninja-build")
    .pip_install("wheel")
    .run_commands(
        # "git clone https://github.com/weedge/vllm.git",
        # "cd vllm &&  pip install -r requirements-cuda.txt",
    )
    .pip_install("vllm", extra_index_url="https://download.pytorch.org/whl/cu126")
    .pip_install(
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
)


# Although vLLM will download weights on-demand, we want to cache them if possible. We'll use [Modal Volumes](https://modal.com/docs/guide/volumes),
# which act as a "shared disk" that all Modal Functions can access, for our cache.

vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)

PROFILE_DIR = "/root/vllm_profile"
vllm_profile = modal.Volume.from_name("vllm_profile", create_if_missing=True)

vllm_image = vllm_image.env(
    {
        "VLLM_USE_V1": "1",
        "VLLM_TORCH_PROFILER_DIR": PROFILE_DIR,
        "LLM_MODEL": os.getenv("LLM_MODEL", "Skywork/Skywork-R1V3-38B"),
        "TP": TP,
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "TORCH_CUDA_ARCH_LIST": "8.0 8.9 9.0+PTX",
    }
)
if BACKEND == "flashinfer":
    vllm_image = vllm_image.pip_install(
        f"flashinfer-python==0.2.2.post1",  # FlashInfer 0.2.3+ does not support per-request generators
        extra_index_url="https://flashinfer.ai/whl/cu126/torch2.6",
    )

if APP_NAME == "achatbot":
    vllm_image = vllm_image.pip_install(
        f"achatbot==0.0.21.post2",
        extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
    )

app = modal.App("vllm-vision-skyworkr1v")

IMAGE_GPU = os.getenv("IMAGE_GPU", None)


with vllm_image.imports():
    from PIL import Image
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, AsyncLLMEngine, AsyncEngineArgs, SamplingParams


@app.function(
    gpu=IMAGE_GPU,
    cpu=2.0,
    retries=0,
    image=vllm_image,
    volumes={
        "/root/.cache/vllm": vllm_cache_vol,
        PROFILE_DIR: vllm_profile,
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_dir,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
)
async def run(func, thinking):
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    gpu_prop = None
    if torch.cuda.is_available():
        gpu_prop = torch.cuda.get_device_properties("cuda")
    print(gpu_prop)

    if asyncio.iscoroutinefunction(func):
        return await func(gpu_prop, thinking)
    else:
        return func(gpu_prop, thinking)


def tokenize(gpu_prop, thinking, image_size=1):
    model_name = os.getenv("LLM_MODEL")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print(f"{tokenizer=}")
    print(f"{tokenizer.eos_token=}")

    prompt = "<image>\n" * image_size + "描述下图片内容"
    messages = [
        {"role": "system", "content": "你是一个中文语音助手, 请用中文回答"},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "图片内容是好的"},
        {"role": "user", "content": prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if thinking is False:
        prompt = prompt.replace("<think>\n", "")
    print(f"{prompt=}")

    return prompt


def generate(gpu_prop, thinking):
    image_file = os.path.join(ASSETS_DIR, "03-Confusing-Pictures.jpg")
    images = [Image.open(image_file)]

    prompt = tokenize(gpu_prop, thinking, len(images))

    MODEL_PATH = os.path.join(HF_MODEL_DIR, os.getenv("LLM_MODEL"))
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=int(os.getenv("TP", "1")),
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 10},
        gpu_memory_utilization=0.7,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=4096,
        repetition_penalty=1.1,
    )
    # https://docs.vllm.ai/en/stable/models/generative_models.html?h=llm.gen#llmgenerate
    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {"image": images},
        },
        sampling_params=sampling_params,
    )
    print(outputs)


async def generate_stream(gpu_prop, thinking):
    image_file = os.path.join(ASSETS_DIR, "03-Confusing-Pictures.jpg")
    images = [Image.open(image_file)]
    prompt = tokenize(gpu_prop, thinking, len(images))

    MODEL_PATH = os.path.join(HF_MODEL_DIR, os.getenv("LLM_MODEL"))
    llm = AsyncLLMEngine.from_engine_args(
        # https://docs.vllm.ai/en/stable/api/vllm/engine/arg_utils.html#vllm.engine.arg_utils.AsyncEngineArgs.__init__
        AsyncEngineArgs(
            model=MODEL_PATH,
            tensor_parallel_size=int(os.getenv("TP", "1")),
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 10},
            gpu_memory_utilization=0.7,
        )
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=4096,
        repetition_penalty=1.1,
    )
    # https://docs.vllm.ai/en/stable/api/vllm/v1/engine/async_llm.html?h=#vllm.v1.engine.async_llm.AsyncLLM.generate
    iterator = llm.generate(
        # https://docs.vllm.ai/en/stable/api/vllm/inputs/data.html#vllm.inputs.data.TextPrompt
        {
            "prompt": prompt,
            "multi_modal_data": {"image": images},
        },
        sampling_params=sampling_params,
        request_id=str(uuid.uuid4()),
    )

    async for part in iterator:
        print(part)


async def achatbot_generate_stream(gpu_prop, thinking):
    from achatbot.core.llm.vllm.vision_skyworkr1v import VllmVisionSkyworkr1v
    from achatbot.types.llm.vllm import VllmEngineArgs
    from achatbot.common.types import SessionCtx
    from achatbot.common.session import Session
    from achatbot.common.logger import Logger

    Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)

    MODEL_PATH = os.path.join(HF_MODEL_DIR, os.getenv("LLM_MODEL"))
    llm = VllmVisionSkyworkr1v(
        **VllmEngineArgs(
            init_chat_prompt="你是一个中文语音智能助手，不要使用特殊字符回复，请使用中文回复。",
            warmup_steps=0,
            chat_history_size=2,
            serv_args=AsyncEngineArgs(
                model=MODEL_PATH,
                tensor_parallel_size=int(os.getenv("TP", "1")),
                trust_remote_code=True,
                limit_mm_per_prompt={"image": 10},
                gpu_memory_utilization=0.7,
            ).__dict__,
        ).__dict__,
    )

    session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)
    url = "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"
    image_bytes = requests.get(url).content
    img = Image.open(io.BytesIO(image_bytes))
    chat_texts = ["这张图片的内容是什么", "你叫什么名字", "讲个故事", "这个是什么时候的图片"]
    for chat_text in chat_texts:
        session.ctx.state["prompt"] = [
            {"type": "image", "image": img},
            {"type": "text", "text": chat_text},
        ]
        # async for result_text in llm.async_generate(session, thinking=thinking):
        async for result_text in llm.async_chat_completion(session, thinking=thinking):
            if result_text is not None:
                print(result_text, flush=True, end="")
    # chat_history size = 2
    print("\n")
    print(llm.get_session_chat_history(session.ctx.client_id))
    img.close()


"""
# 0. download model
modal run src/download_models.py --repo-ids "Skywork/Skywork-R1V3-38B"

# 1. tokenize
modal run src/llm/vllm/vlm/skywork_r1v.py --task tokenize

# NOTE: 
https://huggingface.co/Skywork/Skywork-R1V3-38B/blob/main/config.json
- num_attention_heads: 40
- num_key_value_heads: 8
- Total number of attention heads (40) must be divisible by tensor parallel size
- self.total_num_kv_heads (8) % tp_size == 0

# remove vllm cache
VOLUME_NAME=vllm-cache modal run src/remove_volume_data.py --files "torch_compile_cache"

# 2. generate
IMAGE_GPU=L40s:4 TP=4 modal run src/llm/vllm/vlm/skywork_r1v.py --task generate
IMAGE_GPU=A100:4 TP=4 modal run src/llm/vllm/vlm/skywork_r1v.py --task generate
IMAGE_GPU=A100-80GB:2 TP=2 modal run src/llm/vllm/vlm/skywork_r1v.py --task generate
# use flashinfer
IMAGE_GPU=L40s:4 TP=4 BACKEND=flashinfer modal run src/llm/vllm/vlm/skywork_r1v.py --task generate
IMAGE_GPU=A100:4 TP=4 BACKEND=flashinfer modal run src/llm/vllm/vlm/skywork_r1v.py --task generate
IMAGE_GPU=A100-80GB:2 TP=2 BACKEND=flashinfer modal run src/llm/vllm/vlm/skywork_r1v.py --task generate

# _CudaDeviceProperties(name='NVIDIA H200', major=9, minor=0, total_memory=143156MB, multi_processor_count=132, uuid=91a535d7-4157-249a-65e5-73883afa626e, L2_cache_size=60MB)
IMAGE_GPU=H200 TP=1 modal run src/llm/vllm/vlm/skywork_r1v.py --task generate
IMAGE_GPU=H200 TP=1 BACKEND=flashinfer modal run src/llm/vllm/vlm/skywork_r1v.py --task generate

# NOTE: need update latest torch version to support sm_100 arch
# _CudaDeviceProperties(name='NVIDIA B200', major=10, minor=0, total_memory=182642MB, multi_processor_count=148, uuid=d186e4e8-5421-22be-0d31-ba7ed89e6a79, L2_cache_size=126MB)
IMAGE_GPU=B200 TP=1 modal run src/llm/vllm/vlm/skywork_r1v.py --task generate
IMAGE_GPU=B200 TP=1 BACKEND=flashinfer modal run src/llm/vllm/vlm/skywork_r1v.py --task generate

# 3. generate_stream
IMAGE_GPU=L40s:4 TP=4 modal run src/llm/vllm/vlm/skywork_r1v.py --task generate_stream
IMAGE_GPU=A100:4 TP=4 modal run src/llm/vllm/vlm/skywork_r1v.py --task generate_stream 
IMAGE_GPU=A100-80GB:2 TP=2 modal run src/llm/vllm/vlm/skywork_r1v.py --task generate_stream
# use flashinfer
IMAGE_GPU=L40s:4 TP=4 BACKEND=flashinfer modal run src/llm/vllm/vlm/skywork_r1v.py --task generate_stream
IMAGE_GPU=A100:4 TP=4 BACKEND=flashinfer modal run src/llm/vllm/vlm/skywork_r1v.py --task generate_stream 
IMAGE_GPU=A100-80GB:2 TP=2 BACKEND=flashinfer modal run src/llm/vllm/vlm/skywork_r1v.py --task generate_stream

# 4. achatbot_generate_stream
IMAGE_GPU=L40s:4 TP=4 APP_NAME=achatbot modal run src/llm/vllm/vlm/skywork_r1v.py --task achatbot_generate_stream
IMAGE_GPU=A100:4 TP=4 APP_NAME=achatbot modal run src/llm/vllm/vlm/skywork_r1v.py --task achatbot_generate_stream 
IMAGE_GPU=A100-80GB:2 TP=2 APP_NAME=achatbot modal run src/llm/vllm/vlm/skywork_r1v.py --task achatbot_generate_stream
# use flashinfer
IMAGE_GPU=L40s:4 TP=4 APP_NAME=achatbot BACKEND=flashinfer modal run src/llm/vllm/vlm/skywork_r1v.py --task achatbot_generate_stream
IMAGE_GPU=A100:4 TP=4 APP_NAME=achatbot BACKEND=flashinfer modal run src/llm/vllm/vlm/skywork_r1v.py --task achatbot_generate_stream 
IMAGE_GPU=A100-80GB:2 TP=2 APP_NAME=achatbot BACKEND=flashinfer modal run src/llm/vllm/vlm/skywork_r1v.py --task achatbot_generate_stream
"""


@app.local_entrypoint()
def main(task: str = "tokenize", thinking: Optional[bool] = None):
    print(task, thinking)
    tasks = {
        "tokenize": tokenize,
        "generate": generate,
        "generate_stream": generate_stream,
        "achatbot_generate_stream": achatbot_generate_stream,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(tasks[task], thinking)
