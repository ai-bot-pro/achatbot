import os
import json
import subprocess
import uuid
from pathlib import Path
from typing import Dict
import importlib

import modal
import modal.experimental

from app import model_dir, MODEL_VOL, comfyui_out_dir, COMFYUI_OUT_VOL


# if use nightly, git pull comfyui repo
# # https://github.com/Comfy-Org/ComfyUI-Manager/blob/main/docs/en/v3.38-userdata-security-migration.md
VERSION = os.getenv("VERSION", "0.3.76")

IMAGE_GPU = os.getenv("IMAGE_GPU", "L4")
MAX_INPUTS = int(os.getenv("MAX_INPUTS", "2"))

MODEL_NAME = os.getenv("MODEL_NAME", "flux1_schnell_fp8")
WORKFLOW_CONFIG_DIR = Path(__file__).parent / "workflow_config"
WORKFLOW_CONFIG_PATH = os.getenv(
    "WORKFLOW_CONFIG_PATH", f"{WORKFLOW_CONFIG_DIR}/{MODEL_NAME}_api.json"
)

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
)

image = (
    # https://github.com/Comfy-Org/comfy-cli
    image.uv_pip_install("comfy-cli==1.5.3")  # install comfy-cli
    .run_commands(  # use comfy-cli to install ComfyUI and its dependencies
        # https://github.com/comfyanonymous/ComfyUI
        f"comfy --skip-prompt install --fast-deps --nvidia --version {VERSION}"
    )
    .run_commands(
        # https://github.com/WASasquatch/was-node-suite-comfyui
        "comfy node install --fast-deps was-ns@3.0.1"
    )
    .run_commands(
        "which comfy",
        "comfy --help",
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

image = (
    # copy the app path to the container.
    image.add_local_dir(APP_DIR, f"/root/app", copy=True)
    # copy the ComfyUI workflow JSON to the container.
    .add_local_file(WORKFLOW_CONFIG_PATH, f"/root/{MODEL_NAME}_api.json", copy=True)
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


app = modal.App(name="server", image=image)


@app.cls(
    scaledown_window=300,  # 5 minute container keep alive after it processes an input
    gpu=IMAGE_GPU,
    volumes={
        model_dir: MODEL_VOL,
        comfyui_out_dir: COMFYUI_OUT_VOL,
    },
    timeout=1200,  # default 300s
)
@modal.concurrent(max_inputs=MAX_INPUTS)  # run max inputs per container
class ComfyUI:
    port: int = 8000

    @modal.enter()
    def launch_comfy_background(self):
        # launch the ComfyUI server exactly once when the container starts
        cmd = f"comfy launch --background -- --port {self.port}"
        subprocess.run(cmd, shell=True, check=True)

    @modal.method()
    def infer(self, file_name: str = ""):
        # sometimes the ComfyUI server stops responding (we think because of memory leaks), so this makes sure it's still up
        self.poll_server_health()

        # runs the comfy run --workflow command as a subprocess
        workflow_path = f"{file_name}.json"
        # workflow_path = Path(__file__).parent / f"{os.getenv("MODEL_NAME")}_api.json"
        cmd = f"comfy run --workflow {workflow_path} --wait --timeout 1200 --verbose"
        print(cmd)
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                check=True,
                capture_output=True,
                text=True,  # Decode stdout/stderr as text
            )
            print("ComfyUI stdout:", result.stdout)
            if result.stderr:
                print("ComfyUI stderr:", result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"ComfyUI command failed with exit code {e.returncode}")
            print(f"ComfyUI stdout: {e.stdout}")
            print(f"ComfyUI stderr: {e.stderr}")
            raise  # Re-raise the exception after logging

    @modal.asgi_app()
    def api(self):
        from fastapi import FastAPI, Response
        from fastapi.responses import StreamingResponse

        app = FastAPI()

        @app.post(path="/video")
        def gen_video(item: Dict):
            model_name = item.get("model")
            if model_name is None:
                return Response("Missing model name", status_code=400)

            MODEL_NAME = os.getenv("MODEL_NAME")
            if model_name != MODEL_NAME:
                return Response(f"Model name does not match, now use {MODEL_NAME}", status_code=400)

            file_path = Path(__file__).parent / f"{MODEL_NAME}_api.json"
            images = item.get("images")
            if images is not None and isinstance(images, list) and len(images) > 0:
                file_path = Path(__file__).parent / f"{MODEL_NAME}_img2img_api.json"
            # change workflow conf
            filename = self.change_workflow_conf(file_path, item)
            if filename is None:
                return Response("Failed to change workflow conf", status_code=500)

            # run inference on the currently running container
            # NOTE: need use modal queue to do this
            self.infer.local(filename)

            # returns the video as bytes stream
            # https://fastapi.tiangolo.com/advanced/custom-response/#using-streamingresponse-with-file-like-objects
            def iterfile():  # (1)
                for f in Path(comfyui_out_dir).iterdir():
                    if f.name.startswith(filename):
                        # if f.name.endswith(".mp4"):
                        with open(f, mode="rb") as file_like:  # (2)
                            yield from file_like  # (3)

            def iterfile_lastest():
                latest_mp4 = None
                latest_mtime = 0
                for f in Path(comfyui_out_dir).iterdir():
                    if f.name.endswith(".mp4"):
                        mtime = f.stat().st_mtime
                        if mtime > latest_mtime:
                            latest_mtime = mtime
                            latest_mp4 = f
                if latest_mp4:
                    with open(latest_mp4, mode="rb") as file_like:
                        yield from file_like
                else:
                    print(f"No MP4 file found for filename: {filename}")

            return StreamingResponse(iterfile_lastest(), media_type="video/mp4")

        @app.post(path="/image")
        def gen_image(item: Dict):
            model_name = item.get("model")
            if model_name is None:
                return Response("Missing model name", status_code=400)

            MODEL_NAME = os.getenv("MODEL_NAME")
            if model_name != MODEL_NAME:
                return Response(f"Model name does not match, now use {MODEL_NAME}", status_code=400)

            file_path = Path(__file__).parent / f"{MODEL_NAME}_api.json"
            images = item.get("images")
            if images is not None and isinstance(images, list) and len(images) > 0:
                file_path = Path(__file__).parent / f"{MODEL_NAME}_img2img_api.json"
            # change workflow conf
            filename = self.change_workflow_conf(file_path, item)
            if filename is None:
                return Response("Failed to change workflow conf", status_code=500)

            # run inference on the currently running container
            self.infer.local(filename)

            # returns the image as bytes
            img_bytes = b""
            for f in Path(comfyui_out_dir).iterdir():
                if f.name.startswith(filename):
                    img_bytes = f.read_bytes()

            return Response(img_bytes, media_type="image/jpeg")

        return app

    def change_workflow_conf(self, file_path: Path, item: Dict) -> str:
        try:
            MODEL_NAME = os.getenv("MODEL_NAME")
            module = importlib.import_module(f"app.{MODEL_NAME}")
            func = getattr(module, "change_workflow_conf")
            filename_prefix = func(file_path, **item)
            return filename_prefix
        except ImportError as e:
            print(f"Import error in change_workflow_conf: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred in change_workflow_conf: {e}")
            return None

    def poll_server_health(self) -> None:
        import socket
        import urllib

        try:
            # check if the server is up (response should be immediate)
            req = urllib.request.Request(f"http://127.0.0.1:{self.port}/system_stats")
            urllib.request.urlopen(req, timeout=5)
            print("ComfyUI server is healthy")
        except (socket.timeout, urllib.error.URLError) as e:
            # if no response in 5 seconds, stop the container
            print(f"Server health check failed: {str(e)}")
            modal.experimental.stop_fetching_inputs()

            # all queued inputs will be marked "Failed", so you need to catch these errors in your client and then retry
            raise Exception("ComfyUI server is not healthy, stopping container")


"""
# flux1.0 schnell fp8
modal serve src/comfyui/server.py 
MODEL_NAME=flux1_schnell_fp8 IMAGE_GPU=L40S modal serve src/comfyui/server.py 

# z-image turbo
MODEL_NAME=image_z_image_turbo IMAGE_GPU=L4 modal serve src/comfyui/server.py 
MODEL_NAME=image_z_image_turbo IMAGE_GPU=L40S modal serve src/comfyui/server.py 

# flux2 dev
MODEL_NAME=image_flux2 IMAGE_GPU=L40S modal serve src/comfyui/server.py 

# hunyuan video 1.5 720p t2v
MODEL_NAME=video_hunyuan_video_1_5_720p_t2v IMAGE_GPU=L40S modal serve src/comfyui/server.py

MODEL_NAME=video_fastvideo_wan2_1_i2v_14b_480p_diffusers \
    FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN \
    IMAGE_GPU=L40s:4 modal serve src/comfyui/server.py

MODEL_NAME=video_fastvideo_hunyuan_diffusers \
    FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN \
    IMAGE_GPU=L40s:2 modal serve src/comfyui/server.py
"""
