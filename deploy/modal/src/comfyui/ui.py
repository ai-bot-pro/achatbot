import os
import json
import subprocess
import uuid
from pathlib import Path
from typing import Dict

import modal
import modal.experimental

IMAGE_GPU = os.getenv("IMAGE_GPU", "L4")

image = (  # build up a Modal Image to run ComfyUI, step by step
    modal.Image.debian_slim(  # start from basic Linux with Python
        python_version="3.11"
    )
    .apt_install("git")  # install git to clone ComfyUI
    .uv_pip_install("fastapi[standard]==0.115.4")  # install web dependencies
    .uv_pip_install("comfy-cli==1.5.3")  # install comfy-cli
    .run_commands(  # use comfy-cli to install ComfyUI and its dependencies
        # https://github.com/comfyanonymous/ComfyUI
        "comfy --skip-prompt install --fast-deps --nvidia --version 0.3.75"
    )
    .run_commands(
        "which comfy",
        "comfy --help",
    )
)


# https://github.com/black-forest-labs/flux
# https://huggingface.co/black-forest-labs/FLUX.1-schnell
# https://huggingface.co/Comfy-Org/flux1-schnell/tree/main
# modal run src/download_models.py --repo-ids "Comfy-Org/flux1-schnell" --allow-patterns "flux1-schnell-fp8.safetensors"
from .app import model_dir, MODEL_VOL

# gen_img_dir = "/root/gen_img"
# GEN_IMG_VOL = modal.Volume.from_name("gen_img", create_if_missing=True)


app = modal.App(name="comfyui", image=image)


@app.function(
    max_containers=1,  # limit interactive session to 1 container
    gpu=IMAGE_GPU,
    volumes={
        model_dir: MODEL_VOL,
        gen_img_dir: GEN_IMG_VOL,
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
"""
