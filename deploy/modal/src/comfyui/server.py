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
VERSION = os.getenv("VERSION", "0.3.75")

IMAGE_GPU = os.getenv("IMAGE_GPU", "L4")
MAX_INPUTS = int(os.getenv("MAX_INPUTS", "2"))

MODEL_NAME = os.getenv("MODEL_NAME", "flux1_schnell_fp8")
WORKFLOW_CONFIG_DIR = Path(__file__).parent / "workflow_config"
WORKFLOW_CONFIG_PATH = os.getenv(
    "WORKFLOW_CONFIG_PATH", f"{WORKFLOW_CONFIG_DIR}/{MODEL_NAME}_api.json"
)

APP_DIR = Path(__file__).parent / "app"

image = (  # build up a Modal Image to run ComfyUI, step by step
    modal.Image.debian_slim(  # start from basic Linux with Python
        python_version="3.11"
    )
    .apt_install("git")  # install git to clone ComfyUI
    .uv_pip_install("fastapi[standard]==0.115.4")  # install web dependencies
    # copy the app path to the container.
    .add_local_dir(APP_DIR, f"/root/app", copy=True)
    # copy the ComfyUI workflow JSON to the container.
    .add_local_file(
        WORKFLOW_CONFIG_PATH,
        f"/root/{MODEL_NAME}_api.json",
        copy=True,
    )
    .env(
        {
            "MODEL_NAME": MODEL_NAME,
        }
    )
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

    @modal.fastapi_endpoint(method="POST", path="/video")
    def gen_video(self, item: Dict):
        from fastapi import Response
        from fastapi.responses import StreamingResponse

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

        # returns the video as bytes stream
        # https://fastapi.tiangolo.com/advanced/custom-response/#using-streamingresponse-with-file-like-objects
        def iterfile():  # (1)
            for f in Path(comfyui_out_dir).iterdir():
                if f.name.startswith(filename):
                    with open(f, mode="rb") as file_like:  # (2)
                        yield from file_like  # (3)

        return StreamingResponse(iterfile(), media_type="video/mp4")

    @modal.fastapi_endpoint(method="POST", path="/image")
    def gen_image(self, item: Dict):
        from fastapi import Response

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
"""
