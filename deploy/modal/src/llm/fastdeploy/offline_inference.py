# author: weedge (weege007@gmail.com)

import os
import subprocess
from threading import Thread
from time import perf_counter
from typing import Optional

import modal


LLM_MODEL = os.getenv("LLM_MODEL", "baidu/ERNIE-4.5-0.3B-Paddle")
IMAGE_GPU = os.getenv("IMAGE_GPU", "A100")
FASTDEPLOY_VERSION = os.getenv("FASTDEPLOY_VERSION", "stable")  # stable, nightly
GPU_ARCHS = os.getenv("GPU_ARCHS", "80_90")  # 80_90, 86_89
app = modal.App("fastdeploy-offline-inference")
img = (
    # use openai triton
    # https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/get_started/installation/nvidia_gpu.md
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        # "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy-cuda-12.6:2.0.0",
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "clang", "cmake", "ninja-build")
    .pip_install(
        "paddlepaddle-gpu==3.1.0",
        index_url=" https://www.paddlepaddle.org.cn/packages/stable/cu126/",
    )
    .run_commands(
        f"python -m pip install fastdeploy-gpu -i https://www.paddlepaddle.org.cn/packages/{FASTDEPLOY_VERSION}/fastdeploy-gpu-{GPU_ARCHS}/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
    )
    .env(
        {
            # "TQDM_DISABLE": "1",
            "LLM_MODEL": LLM_MODEL,
            "IMAGE_GPU": IMAGE_GPU,
        }
    )
)

HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)
VIDEO_OUTPUT_DIR = "/gen_video"
video_out_vol = modal.Volume.from_name("gen_video", create_if_missing=True)


@app.function(
    gpu=os.getenv("IMAGE_GPU", "A100"),
    cpu=2.0,
    retries=0,
    image=img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_dir,
        VIDEO_OUTPUT_DIR: video_out_vol,
    },
    timeout=86400,  # default 300s
    max_containers=1,
)
def run(task):
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)

    task()


def check():
    import paddle
    from paddle.jit.marker import unified

    print(paddle.device.get_device())  # 输出类似于'gpu:0'

    # Verify GPU availability
    paddle.utils.run_check()
    # Verify FastDeploy custom operators compilation
    from fastdeploy.model_executor.ops.gpu import beam_search_softmax


def generate():
    from fastdeploy import LLM, SamplingParams

    prompts = [
        "把李白的静夜思改写为现代诗",
        "Write me a poem about large language model.",
    ]

    # Sampling parameters
    # https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/offline_inference.md#24-fastdeploysamplingparams
    sampling_params = SamplingParams(top_p=0.95, max_tokens=6400)

    # Load model
    LLM_MODEL = os.getenv("LLM_MODEL")
    model_path = os.path.join(HF_MODEL_DIR, LLM_MODEL)
    # https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/parameters.md
    llm = LLM(model=model_path, tensor_parallel_size=1, max_model_len=8192)

    # Batch inference (internal request queuing and dynamic batching)
    outputs = llm.generate(prompts, sampling_params)

    # Output results
    for output in outputs:
        print(output)
        prompt = output.prompt
        generated_text = output.outputs.text


def chat():
    from fastdeploy import LLM, SamplingParams

    msg1 = [
        {"role": "system", "content": "I'm a helpful AI assistant."},
        {"role": "user", "content": "把李白的静夜思改写为现代诗"},
    ]
    msg2 = [
        {"role": "system", "content": "I'm a helpful AI assistant."},
        {"role": "user", "content": "Write me a poem about large language model."},
    ]
    messages = [msg1, msg2]

    # Sampling parameters
    # https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/offline_inference.md#24-fastdeploysamplingparams
    sampling_params = SamplingParams(top_p=0.95, max_tokens=6400)

    # Load model
    LLM_MODEL = os.getenv("LLM_MODEL")
    model_path = os.path.join(HF_MODEL_DIR, LLM_MODEL)
    # https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/parameters.md
    llm = LLM(model=model_path, tensor_parallel_size=1, max_model_len=8192)
    # Batch inference (internal request queuing and dynamic batching)
    outputs = llm.chat(messages, sampling_params)

    # Output results
    for output in outputs:
        print(output)
        prompt = output.prompt
        generated_text = output.outputs.text


"""
# https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/offline_inference.md
# https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/offline_inference.md#24-fastdeploysamplingparams
# https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/parameters.md

# default 80_90 GPU ARCH use A100
modal run src/llm/fastdeploy/offline_inference.py
modal run src/llm/fastdeploy/offline_inference.py --task generate
modal run src/llm/fastdeploy/offline_inference.py --task chat

# 86_89 GPU ARCH use L4
GPU_ARCHS=86_89 IMAGE_GPU=L4 modal run src/llm/fastdeploy/offline_inference.py
GPU_ARCHS=86_89 IMAGE_GPU=L4 modal run src/llm/fastdeploy/offline_inference.py --task generate
GPU_ARCHS=86_89 IMAGE_GPU=L4 modal run src/llm/fastdeploy/offline_inference.py --task chat 
"""


@app.local_entrypoint()
def main(task: str = "check"):
    tasks = {
        "check": check,
        "generate": generate,
        "chat": chat,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(tasks[task])
