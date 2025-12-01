import subprocess

import modal

model_dir = "/root/models"
MODEL_VOL = modal.Volume.from_name("models", create_if_missing=True)

# completed workflows write output images/video/audio to this directory
comfyui_out_dir = "/root/comfy/ComfyUI/output"
COMFYUI_OUT_VOL = modal.Volume.from_name("comfyui_output", create_if_missing=True)


# https://docs.comfy.org/installation/manual_install#example-structure
def clear_comfyui_output_dir():
    # clear output file
    subprocess.run(
        f"rm -f {comfyui_out_dir}/_output_images_will_be_put_here", shell=True, check=True
    )
