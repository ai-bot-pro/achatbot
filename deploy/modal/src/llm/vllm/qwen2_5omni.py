import math
import os
import modal

app = modal.App("vllm-generate")

vllm_image = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git",
        "git-lfs",
        "ffmpeg",
        "software-properties-common",
        "libsndfile1",
        "wget",
    )
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        "accelerate",
        "torchdiffeq",
        "x_transformers",
        "setuptools_scm",
        "resampy",
        "qwen-omni-utils",
    )
    .run_commands(
        "wget https://github.com/Kitware/CMake/releases/download/v3.26.1/cmake-3.26.1-Linux-x86_64.sh \
    -q -O /tmp/cmake-install.sh \
    && chmod u+x /tmp/cmake-install.sh \
    && mkdir /opt/cmake-3.26.1 \
    && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.26.1 \
    && rm /tmp/cmake-install.sh \
    && ln -s /opt/cmake-3.26.1/bin/* /usr/local/bin"
    )
    .run_commands(
        "git lfs install",
        "git clone -b new_qwen2_omni_public https://github.com/ai-bot-pro/vllm.git",
        "cd /vllm && git checkout 50952d6e2b954063a7cfee9cb436aa57db065738",
        "cd /vllm && pip install -r requirements/cuda.txt",
        "cd /vllm && pip install . --no-build-isolation",  # u can see a little film
    )
    .run_commands(
        "pip install git+https://github.com/BakerBunker/transformers@21dbefaa54e5bf180464696aa70af0bfc7a61d53",
    )
    .pip_install(
        "flashinfer-python==0.2.0.post2",
        extra_index_url="https://flashinfer.ai/whl/cu121/torch2.6/",
    )
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TORCH_CUDA_ARCH_LIST": "7.5 8.0 8.6 8.7 8.9 9.0",
        }
    )  # faster model transfers
)


# PP need close v1
vllm_image = vllm_image.env(
    {
        "VLLM_USE_V1": os.getenv("VLLM_USE_V1", "0"),
    }
).run_commands(
    "rm -rf /vllm && git clone -b new_qwen2_omni_public https://github.com/ai-bot-pro/vllm.git",
    "cd /vllm && git checkout 84b00e332c5005f59215865120822480b6c0fa2d",
)

HF_MODEL_DIR = "/root/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
VLLM_CACHE_DIR = "/root/.cache/vllm"
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
ASSETS_DIR = "/root/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)


with vllm_image.imports():
    import subprocess
    import torch

    device_count = torch.cuda.device_count()
    devices = ",".join([f"{i}" for i in range(device_count)])
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    if device_count > 1:
        subprocess.run("nvidia-smi topo -m", shell=True, env=os.environ)


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    cpu=2.0,
    retries=0,
    image=vllm_image,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        VLLM_CACHE_DIR: vllm_cache_vol,
        ASSETS_DIR: assets_dir,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=100,
)
def run(func, thinker_gpu_memory_utilization, talker_gpu_memory_utilization, other_cmd_args):
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    gpu_prop = None
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            gpu_prop = torch.cuda.get_device_properties(f"cuda:{i}")
            print(gpu_prop)
    else:
        print("CUDA is not available.")

    func(
        thinker_gpu_memory_utilization=thinker_gpu_memory_utilization,
        talker_gpu_memory_utilization=talker_gpu_memory_utilization,
        other_cmd_args=other_cmd_args,
    )


def thinker_only(**kwargs):
    """
    Only use the Thinker model to generate text.

    multi-modal data text/image/audio/video -> thinker -> text
    """
    thinker_gpu_memory_utilization = kwargs.get("thinker_gpu_memory_utilization", 0.8)
    device_count = torch.cuda.device_count()
    print(f"CUDA device count: {device_count}")
    thinker_devices = ",".join([f"{i}" for i in range(device_count)])
    model_dir = os.path.join(HF_MODEL_DIR, "Qwen/Qwen2.5-Omni-7B")
    cmd = f"python end2end.py --model {model_dir} --prompt audio-in-video-v2 --enforce-eager --thinker-only --thinker-gpu-memory-utilization {thinker_gpu_memory_utilization} --thinker-devices [{thinker_devices}]"
    print(cmd)
    subprocess.run(
        cmd, shell=True, cwd="/vllm/examples/offline_inference/qwen2_5_omni/", env=os.environ
    )


def thinker2talker2wav(**kwargs):
    """
    use thinker to generate text and then use talker to generate audio vq indices code, finally convert audio vq indices code to audio waveform

    multi-modal data text/image/audio/video --> thinker -> text | talker -> audio vq indices code -> code2wav -> audio waveform
    """
    # default l40s
    thinker_gpu_memory_utilization = kwargs.get("thinker_gpu_memory_utilization", 0.6)
    talker_gpu_memory_utilization = kwargs.get("talker_gpu_memory_utilization", 0.3)
    device_count = torch.cuda.device_count()
    print(f"CUDA device count: {device_count}")
    thinker_devices = talker_devices = code2wav_devices = "0"
    if device_count > 1:
        thinker_device_count = math.ceil(device_count / 2)
        thinker_devices = ",".join([f"{i}" for i in range(thinker_device_count)])
        talker_devices = ",".join([f"{i}" for i in range(thinker_device_count, device_count)])
        code2wav_devices = f"{device_count-1}"
    if device_count == 2:
        thinker_devices = "0,1"
    model_dir = os.path.join(HF_MODEL_DIR, "Qwen/Qwen2.5-Omni-7B")
    cmd = f"python end2end.py --model {model_dir} --prompt audio-in-video-v2 --enforce-eager --do-wave --voice-type Chelsie --warmup-voice-type Chelsie --thinker-devices [{thinker_devices}] --thinker-gpu-memory-utilization {thinker_gpu_memory_utilization} --talker-devices [{talker_devices}] --talker-gpu-memory-utilization {talker_gpu_memory_utilization} --code2wav-devices [{code2wav_devices}] --output-dir {ASSETS_DIR}"
    print(cmd)
    subprocess.run(
        cmd, shell=True, cwd="/vllm/examples/offline_inference/qwen2_5_omni/", env=os.environ
    )


def code2wav(**kwargs):
    """
    vq code --> cfm dit -> mel --> bigvgan -> waveforms streaming
    """

    other_cmd_args = kwargs.get("other_cmd_args", "")
    model_dir = os.path.join(HF_MODEL_DIR, "Qwen/Qwen2.5-Omni-7B")
    code_file = os.path.join(ASSETS_DIR, "code2wav.json")
    cmd = f"python code2wav.py --code2wav-model {model_dir} --input-json {code_file} --output-dir {ASSETS_DIR} {other_cmd_args}"
    print(cmd)
    subprocess.run(
        cmd, shell=True, cwd="/vllm/examples/offline_inference/qwen2_5_omni/", env=os.environ
    )


"""
# NOTE: 
# - thinker LM: model weights take 16.73GiB; non_torch_memory takes 0.09GiB; PyTorch activation peak memory takes 5.48GiB; the rest of the memory reserved for KV Cache, so the total memory reserved for the model is 22.3 GiB. must thinker-gpu-memory-utilization * total_gpu_memory > 22.3 GiB
# - talker LM: model weights take 2.55GiB; non_torch_memory takes 0.08GiB; PyTorch activation peak memory takes 4.36GiB; the rest of the memory reserved for KV Cache, so the total memory reserved for the model is 6.9 GiB. must talker-gpu-memory-utilization * total_gpu_memory > 6.9 GiB

IMAGE_GPU=L40s modal run src/llm/vllm/qwen2_5omni.py --task thinker_only
# use tp
IMAGE_GPU=L4:2 modal run src/llm/vllm/qwen2_5omni.py --task thinker_only


IMAGE_GPU=L40s modal run src/llm/vllm/qwen2_5omni.py --task thinker2talker2wav
IMAGE_GPU=L40s:2 modal run src/llm/vllm/qwen2_5omni.py --task thinker2talker2wav --thinker-gpu-memory-utilization 0.9 --talker-gpu-memory-utilization 0.7

# slow with no torch compile
IMAGE_GPU=T4 modal run src/llm/vllm/qwen2_5omni.py --task code2wav
IMAGE_GPU=L4 modal run src/llm/vllm/qwen2_5omni.py --task code2wav

# fast with torch compile
IMAGE_GPU=L4 modal run src/llm/vllm/qwen2_5omni.py --task code2wav --other-cmd-args "--enable-torch-compile"
IMAGE_GPU=L40s modal run src/llm/vllm/qwen2_5omni.py --task code2wav --other-cmd-args "--enable-torch-compile"
IMAGE_GPU=L4 modal run src/llm/vllm/qwen2_5omni.py --task code2wav --other-cmd-args "--enable-torch-compile --odeint-method euler"
IMAGE_GPU=L4 modal run src/llm/vllm/qwen2_5omni.py --task code2wav --other-cmd-args "--enable-torch-compile --multi-waveforms"
"""


@app.local_entrypoint()
def main(
    task: str = "thinker_only",
    thinker_gpu_memory_utilization: str = "0.6",  # thinker-gpu-memory-utilization * total_gpu_memory > 22.3GB
    talker_gpu_memory_utilization: str = "0.3",  # talker-gpu-memory-utilization * total_gpu_memory  > 6.9 GB
    other_cmd_args: str = "",
):
    tasks = {
        "thinker_only": thinker_only,
        "thinker2talker2wav": thinker2talker2wav,
        "code2wav": code2wav,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(
        tasks[task], thinker_gpu_memory_utilization, talker_gpu_memory_utilization, other_cmd_args
    )
