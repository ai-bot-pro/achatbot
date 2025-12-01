# https://docs.comfy.org/tutorials/image/z-image/z-image-turbo
"""
📂 ComfyUI/
├── 📂 models/
│   ├── 📂 text_encoders/
│   │      └── qwen_3_4b.safetensors
│   ├── 📂 diffusion_models/
│   │      └── z_image_turbo_bf16.safetensors
│   └── 📂 vae/
│          └── ae.safetensors
"""

import os
import uuid
import json
import logging
from pathlib import Path
import subprocess

import modal

from . import model_dir, MODEL_VOL, comfyui_out_dir, COMFYUI_OUT_VOL, clear_comfyui_output_dir


# https://github.com/Tongyi-MAI/Z-Image
# https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
# https://huggingface.co/Comfy-Org/z_image_turbo/tree/main/split_files
# modal run src/download_models.py --repo-ids "Comfy-Org/z_image_turbo" --allow-patterns "*.safetensors"
# modal run src/download_models.py --repo-ids "bdsqlsz/qinglong_DetailedEyes_Z-Image" --allow-patterns "qinglong_detailedeye_z-imageV2(comfy).safetensors"


def link_comfyui_dir():
    """
    ymlink the model to the right ComfyUI directory
    NOTE: need link ckpt file, don't to link models dir
    """
    clear_comfyui_output_dir()

    models_dir = f"{model_dir}/Comfy-Org/z_image_turbo/split_files"
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

    # lora ckpt
    lora_ckpt_file = f"{model_dir}/bdsqlsz/qinglong_DetailedEyes_Z-Image/qinglong_detailedeye_z-imageV2(comfy).safetensors"
    cmd = f"ln -s '{lora_ckpt_file}' '/root/comfy/ComfyUI/models/loras/qinglong_detailedeye_z-imageV2(comfy).safetensors'"
    print(cmd)
    subprocess.run(
        cmd,
        shell=True,
        check=True,
    )


def change_workflow_conf(file_path: Path, **kwargs) -> str:
    workflow_data = json.loads(file_path.read_text())
    print(file_path, workflow_data)

    # insert the prompt
    if kwargs.get("prompt"):
        workflow_data["45"]["inputs"]["text"] = kwargs.get("prompt")

    client_id = uuid.uuid4().hex
    workflow_data["58"]["inputs"]["filename_prefix"] = client_id

    # change output image size: width x height
    if kwargs.get("width"):
        workflow_data["41"]["inputs"]["width"] = kwargs.get("width")
    if kwargs.get("height"):
        workflow_data["41"]["inputs"]["height"] = kwargs.get("height")

    # ksample steps
    if kwargs.get("steps"):
        workflow_data["44"]["inputs"]["steps"] = kwargs.get("steps")

    new_workflow_file = f"{client_id}.json"
    json.dump(workflow_data, Path(new_workflow_file).open("w", encoding="utf-8"))

    return client_id
