import os
import json
import subprocess
import uuid
from pathlib import Path
from typing import Dict
import importlib

import modal
import modal.experimental

IMAGE_GPU = os.getenv("IMAGE_GPU", "L4")
MAX_INPUTS = int(os.getenv("MAX_INPUTS", "2"))

MODULE_NAME = os.getenv("MODULE_NAME", "app.flux1_schnell_fp8")
APP_DIR = Path(__file__).parent / "app"
WORKFLOW_CONFIG_DIR = Path(__file__).parent / "workflow_config"
WORKFLOW_CONFIG_PATH = os.getenv(
    "WORKFLOW_CONFIG_PATH", f"{WORKFLOW_CONFIG_DIR}/flux1-schnell-fp8_workflow_api.json"
)
CONFIG_NAME = WORKFLOW_CONFIG_PATH.split("/")[-1]
# print(WORKFLOW_CONFIG_PATH, CONFIG_NAME)

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
        # https://github.com/WASasquatch/was-node-suite-comfyui
        "comfy node install --fast-deps was-ns@3.0.1"
    )
    # copy the app path to the container.
    .add_local_dir(APP_DIR, f"/root/app", copy=True)
    # copy the ComfyUI workflow JSON to the container.
    .add_local_file(
        WORKFLOW_CONFIG_PATH,
        f"/root/{CONFIG_NAME}",
        copy=True,
    )
    .env(
        {
            "CONFIG_NAME": CONFIG_NAME,
            "MODULE_NAME": MODULE_NAME,
        }
    )
)


# https://github.com/black-forest-labs/flux
# https://huggingface.co/black-forest-labs/FLUX.1-schnell
# https://huggingface.co/Comfy-Org/flux1-schnell/tree/main
# modal run src/download_models.py --repo-ids "Comfy-Org/flux1-schnell" --allow-patterns "flux1-schnell-fp8.safetensors"
model_dir = "/root/models"
MODEL_VOL = modal.Volume.from_name("models", create_if_missing=True)

# completed workflows write output images/video/audio to this directory
comfyui_out_dir = "/root/comfy/ComfyUI/output"
COMFYUI_OUT_VOL = modal.Volume.from_name("comfyui_output", create_if_missing=True)


# https://docs.comfy.org/installation/manual_install#example-structure
def link_comfyui_dir():
    try:
        MODULE_NAME = os.getenv("MODULE_NAME")
        module = importlib.import_module(f"{MODULE_NAME}")
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
    def infer(self, workflow_path: str = "/root/workflow_api.json"):
        # sometimes the ComfyUI server stops responding (we think because of memory leaks), so this makes sure it's still up
        self.poll_server_health()

        # runs the comfy run --workflow command as a subprocess
        cmd = f"comfy run --workflow {workflow_path} --wait --timeout 1200 --verbose"
        subprocess.run(cmd, shell=True, check=True)

        # looks up the name of the output image file based on the workflow
        workflow = json.loads(Path(workflow_path).read_text())
        file_prefix = [
            node.get("inputs")
            for node in workflow.values()
            if node.get("class_type") == "SaveImage"
        ][0]["filename_prefix"]

        # returns the image as bytes
        for f in Path(comfyui_out_dir).iterdir():
            if f.name.startswith(file_prefix):
                return f.read_bytes()

    @modal.fastapi_endpoint(method="POST")
    def api(self, item: Dict):
        from fastapi import Response

        CONFIG_NAME = os.getenv("CONFIG_NAME")
        workflow_data = json.loads((Path(__file__).parent / f"{CONFIG_NAME}").read_text())

        # insert the prompt
        workflow_data["6"]["inputs"]["text"] = item["prompt"]

        # give the output image a unique id per client request
        client_id = uuid.uuid4().hex
        workflow_data["9"]["inputs"]["filename_prefix"] = client_id

        # save this updated workflow to a new file
        new_workflow_file = f"{client_id}.json"
        json.dump(workflow_data, Path(new_workflow_file).open("w"))

        # run inference on the currently running container
        img_bytes = self.infer.local(new_workflow_file)

        return Response(img_bytes, media_type="image/jpeg")

    def poll_server_health(self) -> Dict:
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
modal serve src/comfyui/server.py 
IMAGE_GPU=L40S modal serve src/comfyui/server.py 
"""
