import os
import json
import subprocess
import importlib
import uuid
from pathlib import Path
from typing import Dict

import modal
import modal.experimental

from app import model_dir, MODEL_VOL, comfyui_out_dir, COMFYUI_OUT_VOL


# if use nightly, git pull comfyui repo
# https://github.com/Comfy-Org/ComfyUI-Manager/blob/main/docs/en/v3.38-userdata-security-migration.md
VERSION = os.getenv("VERSION", "0.3.76")
IMAGE_GPU = os.getenv("IMAGE_GPU", "L4")
MODEL_NAME = os.getenv("MODEL_NAME", "flux1_schnell_fp8")

# https://hao-ai-lab.github.io/FastVideo/inference/optimizations/?h=fastvideo_attention_backend#available-backends
FASTVIDEO_ATTENTION_BACKEND = os.getenv("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")

APP_DIR = Path(__file__).parent / "app"
image = (  # build up a Modal Image to run ComfyUI, step by step
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git")  # install git to clone ComfyUI
    .uv_pip_install("fastapi[standard]==0.115.4")  # install web dependencies
    # https://github.com/Comfy-Org/comfy-cli (manage comfyui and model)
    .uv_pip_install("comfy-cli==1.5.3")  # install comfy-cli
    .run_commands(  # use comfy-cli to install ComfyUI and its dependencies
        # https://github.com/comfyanonymous/ComfyUI
        f"comfy --skip-prompt install --fast-deps --nvidia --version {VERSION}"
    )
    .run_commands(
        "which comfy",
        "comfy --help",
    )
    .env(
        {
            "MODEL_NAME": MODEL_NAME,
        }
    )
)

if "fastvideo" in MODEL_NAME:
    image = (
        image.uv_pip_install("fastvideo", "torchaudio")
        .apt_install("ffmpeg")
        # NOTE: use comfyUI manager to install custom node FastVideo
        .run_commands(
            "git clone https://github.com/weedge/FastVideo.git",
            "cp -r /FastVideo/comfyui /root/comfy/ComfyUI/custom_nodes/FastVideo",
        )
        .run_commands(
            "cd /root/comfy/ComfyUI/custom_nodes && git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
        )
        .env(
            {
                "FASTVIDEO_ATTENTION_BACKEND": FASTVIDEO_ATTENTION_BACKEND,
            }
        )
    )

    if "FLASH_ATTN" in FASTVIDEO_ATTENTION_BACKEND:
        # NOTE:Cannot use FlashAttention-2 backend for dtype other than torch.float16 or torch.bfloat16.
        image = image.run_commands("pip install flash-attn==2.7.4.post1 --no-build-isolation")
    if "SLIDING_TILE_ATTN" in FASTVIDEO_ATTENTION_BACKEND:
        image = image.uv_pip_install("st_attn==0.0.4")
    if "VIDEO_SPARSE_ATTN" in FASTVIDEO_ATTENTION_BACKEND:
        image = image.run_commands(
            "cd /FastVideo && git submodule update --init --recursive",
            "cd /FastVideo && python setup_vsa.py install",
        )
    if "SAGE_ATTN" in FASTVIDEO_ATTENTION_BACKEND:
        image = image.run_commands(
            "git clone https://github.com/thu-ml/SageAttention.git",
            "cd /SageAttention && pip install -e .",
        )
    if "SAGE_ATTN_THREE" in FASTVIDEO_ATTENTION_BACKEND:
        # need python>=3.13, torch>=2.8.0, CUDA >=12.8
        image = image.run_commands(
            "git clone https://huggingface.co/jt-zhang/SageAttention3",
            "cp /SageAttention3/setup.py /FastVideo/fastvideo/attention/backends/",
            "cp -r /SageAttention3/sageattn /FastVideo/fastvideo/attention/backends/",
            "cd /FastVideo/fastvideo/attention/backends/ && python setup.py install",
        )

# copy the app path to the container.
image = image.add_local_dir(APP_DIR, f"/root/app", copy=True)


# https://docs.comfy.org/installation/manual_install#example-structure
def link_comfyui_dir():
    try:
        MODEL_NAME = os.getenv("MODEL_NAME")
        module = importlib.import_module(f"app.{MODEL_NAME}")
        print(module)
        func = getattr(module, "link_comfyui_dir")
        func()
    except ImportError as e:
        print(e)
    except Exception as e:
        print(e)


image = image.run_function(
    link_comfyui_dir,
    volumes={model_dir: MODEL_VOL},
)


app = modal.App(name="comfyui", image=image)


@app.function(
    max_containers=1,  # limit interactive session to 1 container
    memory="32768",  # 32GB
    cpu=8.0,  # 8 core
    gpu=IMAGE_GPU,
    volumes={
        model_dir: MODEL_VOL,
        comfyui_out_dir: COMFYUI_OUT_VOL,
    },
)
@modal.concurrent(
    max_inputs=10
)  # required for UI startup process which runs several API calls concurrently
@modal.web_server(8000, startup_timeout=60)
def ui():
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)


"""
modal serve src/comfyui/ui.py 
IMAGE_GPU=L40S modal serve src/comfyui/ui.py 

MODEL_NAME=image_z_image_turbo modal serve src/comfyui/ui.py 
MODEL_NAME=image_z_image_turbo IMAGE_GPU=L40S modal serve src/comfyui/ui.py 

MODEL_NAME=image_flux2 IMAGE_GPU=L40S modal serve src/comfyui/ui.py 

# hunyuan video 1.5
MODEL_NAME=video_hunyuan_video_1.5_720p_t2v IMAGE_GPU=L40S modal serve src/comfyui/ui.py 

MODEL_NAME=video_fastvideo_wan2_1_i2v_14b_480p_diffusers IMAGE_GPU=L40s:4 modal serve src/comfyui/ui.py 
MODEL_NAME=video_fastvideo_wan2_1_i2v_14b_480p_diffusers \
    FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN \
    IMAGE_GPU=L40s:4 modal serve src/comfyui/ui.py 

MODEL_NAME=video_fastvideo_hunyuan_diffusers IMAGE_GPU=L40s:2 modal serve src/comfyui/ui.py 
MODEL_NAME=video_fastvideo_hunyuan_diffusers \
    FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN \
    IMAGE_GPU=L40s:2 modal serve src/comfyui/ui.py 
"""
