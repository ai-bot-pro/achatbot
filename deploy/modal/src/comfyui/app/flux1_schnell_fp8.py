import os
import uuid
import json
import logging
from pathlib import Path
import subprocess

import modal

from . import model_dir, MODEL_VOL, comfyui_out_dir, COMFYUI_OUT_VOL


logger = logging.getLogger(__name__)

# https://github.com/black-forest-labs/flux
# https://huggingface.co/black-forest-labs/FLUX.1-schnell
# https://huggingface.co/Comfy-Org/flux1-schnell/tree/main
# modal run src/download_models.py --repo-ids "Comfy-Org/flux1-schnell" --allow-patterns "flux1-schnell-fp8.safetensors"


# https://docs.comfy.org/installation/manual_install#example-structure
def link_comfyui_dir():
    # symlink the model to the right ComfyUI directory
    ckpt_path = f"{model_dir}/Comfy-Org/flux1-schnell/flux1-schnell-fp8.safetensors"
    subprocess.run(
        f"ln -s {ckpt_path} /root/comfy/ComfyUI/models/checkpoints/flux1-schnell-fp8.safetensors",
        shell=True,
        check=True,
    )

    # clear output file
    subprocess.run(
        f"rm -f {comfyui_out_dir}/_output_images_will_be_put_here", shell=True, check=True
    )


def change_workflow_conf(file_path: Path, **kwargs):
    workflow_data = json.loads(file_path.read_text())

    # insert the prompt
    if kwargs.get("prompt"):
        workflow_data["6"]["inputs"]["text"] = kwargs.get("prompt")

    # give the output image a unique id per client request
    client_id = uuid.uuid4().hex
    workflow_data["9"]["inputs"]["filename_prefix"] = client_id

    # change output image size: width x height
    if kwargs.get("width"):
        workflow_data["27"]["inputs"]["width"] = kwargs.get("width")
    if kwargs.get("height"):
        workflow_data["27"]["inputs"]["height"] = kwargs.get("height")

    # ksample steps
    if kwargs.get("steps"):
        workflow_data["31"]["inputs"]["steps"] = kwargs.get("steps")

    print(f"{workflow_data=}")

    # save this updated workflow to a new file
    new_workflow_file = f"{client_id}.json"
    json.dump(workflow_data, Path(new_workflow_file).open("w"))

    return new_workflow_file
