import os
import subprocess
import sys

import modal


image = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    # https://hub.docker.com/r/pytorch/pytorch/tags
    modal.Image.from_registry(
        "pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel",
        add_python="3.11",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "clang")
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "torchaudio==2.4.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install("xformers==0.0.28", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("misaki[en]", "ninja", "psutil", "packaging")
    .pip_install("flash_attn", extra_options="--no-build-isolation")
    .run_commands(
        "git clone https://github.com/weedge/MultiTalk.git",
        "cd /MultiTalk && git checkout 7ec3ca546d14050df6a0a46eb16bf07a09745f66",
        "cd /MultiTalk && pip install -r requirements.txt",
    )
    .pip_install("librosa")
    .pip_install("transformers==4.49.0")
    .run_commands(
        "cd /MultiTalk && git checkout 7ec3ca546d14050df6a0a46eb16bf07a09745f66",
    )
)


app = modal.App("meigen_multitalk")

HF_MODEL_DIR = "/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
VIDEO_OUTPUT_DIR = "/gen_video"
video_out_vol = modal.Volume.from_name("gen_video", create_if_missing=True)


@app.function(
    # gpu="T4",
    retries=modal.Retries(initial_delay=0.0, max_retries=1),
    cpu=2.0,
    image=image,
    volumes={HF_MODEL_DIR: hf_model_vol},
    timeout=86400,  # [10,86400]
    max_containers=1,
)
def prepare():
    cmd = f"cp {HF_MODEL_DIR}/MeiGen-AI/MeiGen-MultiTalk/diffusion_pytorch_model.safetensors.index.json {HF_MODEL_DIR}/Wan-AI/Wan2.1-I2V-14B-480P/"
    print(cmd)
    subprocess.run(cmd, shell=True, check=True, cwd="/")
    cmd = f"cp {HF_MODEL_DIR}/MeiGen-AI/MeiGen-MultiTalk/multitalk.safetensors {HF_MODEL_DIR}/Wan-AI/Wan2.1-I2V-14B-480P/"
    print(cmd)
    subprocess.run(cmd, shell=True, check=True, cwd="/")


"""
modal run src/download_models.py --repo-ids "TencentGameMate/chinese-wav2vec2-base" --revision "refs/pr/1"
modal run src/download_models.py --repo-ids "MeiGen-AI/MeiGen-MultiTalk"
modal run src/download_models.py --repo-ids "Wan-AI/Wan2.1-I2V-14B-480P"
#modal run src/download_models.py --repo-ids "hexgrad/Kokoro-82M"

modal run src/video/meigen_multitalk.py::prepare
modal run src/video/meigen_multitalk.py::run
modal run src/video/meigen_multitalk.py::run --mode multi
"""


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L40s"),
    cpu=2.0,
    image=image,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        VIDEO_OUTPUT_DIR: video_out_vol,
    },
    timeout=86400,  # [10,86400]
    max_containers=1,
)
def run(mode="single"):
    """
    mode:
    - single: Single-Persion
    - multi: Multi-Person
    """
    cmd = f"""python generate_multitalk.py \
    --ckpt_dir {HF_MODEL_DIR}/Wan-AI/Wan2.1-I2V-14B-480P \
    --wav2vec_dir {HF_MODEL_DIR}/TencentGameMate/chinese-wav2vec2-base \
    --input_json examples/multitalk_example_1.json \
    --sample_steps 40 \
    --mode streaming \
    --num_persistent_param_in_dit 0 \
    --use_teacache \
    --save_file {VIDEO_OUTPUT_DIR}/single_long_lowvram_exp"""

    if mode == "multi":
        cmd = f"""python generate_multitalk.py \
        --ckpt_dir {HF_MODEL_DIR}/Wan-AI/Wan2.1-I2V-14B-480P \
        --wav2vec_dir {HF_MODEL_DIR}/TencentGameMate/chinese-wav2vec2-base \
        --input_json examples/multitalk_example_2.json \
        --sample_steps 40 \
        --mode streaming \
        --num_persistent_param_in_dit 0 \
        --use_teacache \
        --save_file {VIDEO_OUTPUT_DIR}/multi_long_lowvram_exp"""

    print(cmd)
    subprocess.run(cmd, shell=True, check=True, cwd="/MultiTalk", env=os.environ)

    video_out_vol.commit()
