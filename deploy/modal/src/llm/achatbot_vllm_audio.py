import os
import subprocess
import asyncio

import modal

ASR_TAG = os.getenv("ASR_TAG", "whisper_vllm_asr")

BACKEND = os.getenv("BACKEND", "")
TP = os.getenv("TP", "1")

PROFILE_DIR = "/root/vllm_profile"
vllm_profile = modal.Volume.from_name("vllm_profile", create_if_missing=True)

app = modal.App(f"vllm-run-whisper-achatbot")

image = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "cmake", "ninja-build")
    .pip_install("wheel")
    .run_commands(
        # "git clone https://github.com/weedge/vllm.git",
        # "cd vllm &&  pip install -r requirements-cuda.txt",
    )
    # https://github.com/vllm-project/vllm/issues/19538
    .pip_install("vllm[audio]==0.9.2", extra_index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("huggingface_hub[hf_transfer]", "transformers==4.52.4", "accelerate")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # faster model transfers
            # "VLLM_USE_V1": "1", # VLLM_USE_V1=1 is not supported with ['WhisperForConditionalGeneration']
            "VLLM_TORCH_PROFILER_DIR": PROFILE_DIR,
            "TP": TP,
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "TORCH_CUDA_ARCH_LIST": "8.0 8.9 9.0+PTX",
            "ASR_TAG": ASR_TAG,
        }
    )
    .env(
        {
            "TORCHDYNAMO_VERBOSE": "1",
            # "TORCH_LOGS": "+dynamo",
            "TORCHINDUCTOR_FX_GRAPH_CACHE": 1, # https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html
        }
    )
)
if BACKEND == "flashinfer":
    image = image.pip_install(
        f"flashinfer-python==0.2.2.post1",  # FlashInfer 0.2.3+ does not support per-request generators
        extra_index_url="https://flashinfer.ai/whl/cu126/torch2.6",
    )

# Although vLLM will download weights on-demand, we want to cache them if possible. We'll use [Modal Volumes](https://modal.com/docs/guide/volumes),
# which act as a "shared disk" that all Modal Functions can access, for our cache.
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
MODEL_DIR = "/root/.achatbot/models"
model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)

image = image.pip_install(
    f"achatbot=={os.getenv('ACHATBOT_VERSION', '0.0.24')}",
    extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
)


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    retries=0,
    image=image,
    volumes={
        MODEL_DIR: model_vol,
        ASSETS_DIR: assets_dir,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
)
async def run(
    func: callable = None,
    **kwargs,
) -> None:
    import torch

    print("torch:", torch.__version__)
    print("cuda:", torch.version.cuda)
    print("_GLIBCXX_USE_CXX11_ABI", torch._C._GLIBCXX_USE_CXX11_ABI)

    subprocess.run("pip show transformers", shell=True)
    subprocess.run("pip show vllm", shell=True)
    subprocess.run("pip show achatbot", shell=True)
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("which nvcc", shell=True)
    subprocess.run("nvcc --version", shell=True)

    print(f"{kwargs=}")
    if asyncio.iscoroutinefunction(func):
        await func(**kwargs)
    else:
        func(**kwargs)


async def transcribe(**kwargs):
    from achatbot.common.interface import IAsr
    from achatbot.modules.speech.asr import ASREnvInit
    from achatbot.common.session import Session
    from achatbot.common.types import SessionCtx
    from achatbot.common.logger import Logger

    Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)

    asr: IAsr = ASREnvInit.initASREngine(os.getenv("ASR_TAG", "whisper_vllm_asr"), **kwargs)
    audio_file = os.path.join(ASSETS_DIR, "asr_example_zh.wav")
    session = Session(**SessionCtx("test_client_id", 16000, 2).__dict__)
    asr.set_audio_data(audio_file)
    res = await asr.transcribe(session)
    print(res)


"""
https://huggingface.co/openai/whisper-large-v3-turbo

# download model weights
modal run src/download_models.py --repo-ids "openai/whisper-large-v3"
modal run src/download_models.py --repo-ids "openai/whisper-large-v3-turbo"

# use large-v3-turbo model with vllm
ASR_TAG=whisper_vllm_asr modal run src/llm/achatbot_vllm_audio.py --task transcribe --model-name-or-path "openai/whisper-large-v3-turbo"

# use large-v3-turbo model with torch compile
ASR_TAG=whisper_transformers_torch_compile_asr modal run src/llm/achatbot_vllm_audio.py --task transcribe --model-name-or-path "openai/whisper-large-v3-turbo"

"""


@app.local_entrypoint()
def main(task: str = "transcribe", model_name_or_path: str = "openai/whisper-large-v3-turbo"):
    tasks = {
        "transcribe": transcribe,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    model_name_or_path = os.path.join(MODEL_DIR, model_name_or_path)
    run.remote(
        func=tasks[task],
        model_name_or_path=model_name_or_path,
    )
