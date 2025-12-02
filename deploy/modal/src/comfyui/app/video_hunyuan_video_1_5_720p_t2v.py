# https://docs.comfy.org/tutorials/video/hunyuan/hunyuan-video-1-5
"""
:open_file_folder: ComfyUI/
├── :open_file_folder: models/
│   ├── :open_file_folder: text_encoders/
│   │      ├── qwen_2.5_vl_7b_fp8_scaled.safetensors
│   │      └── byt5_small_glyphxl_fp16.safetensors
│   ├── :open_file_folder: diffusion_models/
│   │      ├── hunyuanvideo1.5_1080p_sr_distilled_fp16.safetensors
│   │      └── hunyuanvideo1.5_720p_t2v_fp16.safetensors
│   └── :open_file_folder: vae/
│          └── hunyuanvideo15_vae_fp16.safetensors
"""

import uuid
import json
import logging
from pathlib import Path
import subprocess

import modal

from . import model_dir, MODEL_VOL, comfyui_out_dir, COMFYUI_OUT_VOL, clear_comfyui_output_dir


# https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5
# https://huggingface.co/tencent/HunyuanVideo-1.5
# https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/tree/main/split_files
# modal run src/download_models.py --repo-ids "Comfy-Org/HunyuanVideo_1.5_repackaged" --allow-patterns "*.safetensors"


def link_comfyui_dir():
    """
    ymlink the model to the right ComfyUI directory
    NOTE: need link ckpt file, don't to link models dir
    """
    clear_comfyui_output_dir()

    models_dir = f"{model_dir}/Comfy-Org/HunyuanVideo_1.5_repackaged/split_files"
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
    client_id = uuid.uuid4().hex

    if kwargs.get("prompt"):
        workflow_data["44"]["inputs"]["text"] = kwargs.get("prompt")

    workflow_data["102"]["inputs"]["filename_prefix"] = client_id
    if kwargs.get("codec"):
        workflow_data["102"]["inputs"]["codec"] = kwargs.get("codec")

    if kwargs.get("width"):
        workflow_data["124"]["inputs"]["width"] = kwargs.get("width")
    if kwargs.get("height"):
        workflow_data["124"]["inputs"]["height"] = kwargs.get("height")
    if kwargs.get("length"):
        workflow_data["124"]["inputs"]["length"] = kwargs.get("length")

    if kwargs.get("steps"):
        workflow_data["128"]["inputs"]["steps"] = kwargs.get("steps")

    print(file_path, workflow_data)
    new_workflow_file = f"{client_id}.json"
    json.dump(workflow_data, Path(new_workflow_file).open("w", encoding="utf-8"))

    return client_id
