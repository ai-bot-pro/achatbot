import subprocess
import modal
import os

app = modal.App("remove_volume_data")
VOLUME_NAME = os.getenv("VOLUME_NAME", "train_output")

# We also define the dependencies for our Function by specifying an
# [Image](https://modal.com/docs/guide/images).

image = modal.Image.debian_slim(python_version="3.11")

VOL_DIR = "/volume_data"
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    # gpu="T4",
    retries=0,
    image=image,
    # secrets=[modal.Secret.from_name("achatbot")],
    volumes={VOL_DIR: vol},
    timeout=1200,
    scaledown_window=1200,
)
def remove_data(files: str) -> str:
    import os

    for _file in files.split(","):
        local_dir = os.path.join(VOL_DIR, _file)
        cmd = f"rm -rf {local_dir}"
        print(f"{cmd}")
        subprocess.run(cmd, shell=True, check=True)


"""
VOLUME_NAME=train_output modal run src/remove_volume_data.py --files "finetune_glm4voice_stage1_20250515092352"
VOLUME_NAME=train_output modal run src/remove_volume_data.py --files "finetune_glm4voice_stage1_20250515092815,finetune_glm4voice_stage1_20250515093726,finetune_glm4voice_stage1_20250515093855,finetune_glm4voice_stage1_20250515100234,finetune_glm4voice_stage1_20250515100902,finetune_glm4voice_stage1_20250515102311,finetune_glm4voice_stage1"

VOLUME_NAME=models modal run src/remove_volume_data.py --files "MeiGen-AI"
"""


@app.local_entrypoint()
def main(files: str):
    remove_data.remote(files)
