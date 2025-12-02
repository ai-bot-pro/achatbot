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
VERSION = os.getenv("VERSION", "0.3.75")
IMAGE_GPU = os.getenv("IMAGE_GPU", "L4")
MODEL_NAME = os.getenv("MODEL_NAME", "flux1_schnell_fp8")

APP_DIR = Path(__file__).parent / "app"
image = (  # build up a Modal Image to run ComfyUI, step by step
    modal.Image.debian_slim(  # start from basic Linux with Python
        python_version="3.11"
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
    # copy the app path to the container.
    .add_local_dir(APP_DIR, f"/root/app", copy=True)
    .env(
        {
            "MODEL_NAME": MODEL_NAME,
        }
    )
)


# https://docs.comfy.org/installation/manual_install#example-structure
def link_comfyui_dir():
    try:
        MODEL_NAME = os.getenv("MODEL_NAME")
        module = importlib.import_module(f"app.{MODEL_NAME}")
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
"""
