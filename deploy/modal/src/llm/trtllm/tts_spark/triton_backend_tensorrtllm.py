# https://nvidia.github.io/TensorRT-LLM/index.html (nice doc)
"""
NeMo -------------
                  |
HuggingFace ------
                  |   convert                             build                    load
Modelopt ---------  ----------> TensorRT-LLM Checkpoint --------> TensorRT Engine ------> TensorRT-LLM ModelRunner
                  |
JAX --------------
                  |
DeepSpeed --------
"""
# https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html

import os
import modal

app = modal.App("be-trtllm")

trtllm_image = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.10",
    )
    .entrypoint([])  # remove verbose logging by base image on entry
    .apt_install(
        "git", "git-lfs", "openmpi-bin", "libopenmpi-dev", "wget"
    )  # OpenMPI for distributed communication
    .pip_install(
        "tensorrt-llm==0.17.0.post1",
        # "pynvml<12",  # avoid breaking change to pynvml version API for tensorrt_llm
        # "tensorrt==10.8.0.43",
        pre=True,
        extra_index_url="https://pypi.nvidia.com",
    )
    .env({})
)


HF_MODEL_DIR = "/root/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
TRT_MODEL_DIR = "/root/trt_models"
trt_model_vol = modal.Volume.from_name("triton_trtllm_models", create_if_missing=True)
