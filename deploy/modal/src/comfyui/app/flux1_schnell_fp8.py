import subprocess
import modal
from . import model_dir, MODEL_VOL, comfyui_out_dir, COMFYUI_OUT_VOL


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
