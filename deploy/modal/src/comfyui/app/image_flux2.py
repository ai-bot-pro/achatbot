# https://docs.comfy.org/tutorials/flux/flux-2-dev
"""
📂 ComfyUI/
├── 📂 models/
│   ├── 📂 text_encoders/
│   │      └── mistral_3_small_flux2_bf16.safetensors
│   │      └── mistral_3_small_flux2_fp8.safetensors
│   ├── 📂 diffusion_models/
│   │      └── flux2_dev_fp8mixed.safetensors
│   └── 📂 vae/
│          └── flux2-vae.safetensors
"""

import os
import uuid
import json
import logging
from pathlib import Path
import subprocess

import modal

from . import model_dir, MODEL_VOL, comfyui_out_dir, COMFYUI_OUT_VOL, clear_comfyui_output_dir


# https://github.com/black-forest-labs/flux2
# https://huggingface.co/black-forest-labs/FLUX.2-dev
# https://huggingface.co/Comfy-Org/flux2-dev/tree/main/split_files
# modal run src/download_models.py --repo-ids "Comfy-Org/flux2-dev" --allow-patterns "*.safetensors"


def link_comfyui_dir():
    """
    ymlink the model to the right ComfyUI directory
    NOTE: need link ckpt file, don't to link models dir
    """
    clear_comfyui_output_dir()

    models_dir = f"{model_dir}/Comfy-Org/flux2-dev/split_files"
    for ckpt_file in Path(models_dir).glob("**/*.safetensors"):
        cmd = (
            f"ln -s {ckpt_file} /root/comfy/ComfyUI/models/{ckpt_file.parent.name}/{ckpt_file.name}"
        )
        print(cmd)
        subprocess.run(
            cmd,
            shell=True,
            check=True,
        )


def change_workflow_conf(file_path: Path, **kwargs) -> str:
    workflow_data = json.loads(file_path.read_text())

    # insert the prompt
    if kwargs.get("prompt"):
        workflow_data["6"]["inputs"]["text"] = kwargs.get("prompt")

    client_id = uuid.uuid4().hex
    workflow_data["9"]["inputs"]["filename_prefix"] = client_id

    # change output image size: width x height
    if kwargs.get("width"):
        workflow_data["47"]["inputs"]["width"] = kwargs.get("width")
        workflow_data["48"]["inputs"]["width"] = kwargs.get("width")
    if kwargs.get("height"):
        workflow_data["47"]["inputs"]["height"] = kwargs.get("height")
        workflow_data["48"]["inputs"]["height"] = kwargs.get("height")

    # ksample steps
    if kwargs.get("steps"):
        workflow_data["48"]["inputs"]["steps"] = kwargs.get("steps")

    images = kwargs.get("images")
    if images is not None and isinstance(images, list) and len(images) > 0:
        # now just support one image input
        if isinstance(images[0], bytes):
            input_image_filename = f"input_{client_id}.jpeg"
            input_image_path = f"{comfyui_out_dir}/{input_image_filename}"
            with open(input_image_path, "wb") as f:
                f.write(images[0])
            workflow_data["46"]["inputs"]["image"] = input_image_path

    new_workflow_file = f"{client_id}.json"
    print(file_path, workflow_data)
    json.dump(workflow_data, Path(new_workflow_file).open("w", encoding="utf-8"))

    return client_id
