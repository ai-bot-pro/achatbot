import modal
import os

achatbot_version = os.getenv("ACHATBOT_VERSION", "0.0.24")
LLM_TAG = os.getenv("LLM_TAG", "llm_ctranslate2_generator")
img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "clang", "cmake")
    .pip_install("wheel")
    .pip_install(
        [
            "achatbot["
            "fastapi_bot_server,"
            "livekit,livekit-api,daily,agora,"
            "silero_vad_analyzer,"
            "sense_voice_asr,deepgram_asr_processor,"
            "tts_edge,"
            "queue"
            f"]=={achatbot_version}",
        ],
        extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
    )
    .pip_install("onnxruntime", "funasr_onnx")
    .env(
        {
            "ACHATBOT_PKG": "1",
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "info"),
            "ACHATBOT_TASK_TYPE": "asyncio",
        }
    )
)

if LLM_TAG == "llm_ctranslate2_generator":
    img = img.pip_install(
        "ctranslate2",
        "transformers[torch]",
    )
if LLM_TAG == "llm_vllm_generator":
    img = img.pip_install(
        "vllm==0.8.0",
        "transformers==4.51.3",
    ).env(
        {
            # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
            "TORCH_CUDA_ARCH_LIST": "8.0 8.6 8.9 9.0 10.0",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "VLLM_USE_V1": os.getenv("VLLM_USE_V1", "1"),
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )
if LLM_TAG == "llm_sglang_generator":
    img = (
        img.pip_install(  # add sglang and some Python dependencies
            # as per sglang website: https://docs.sglang.ai/get_started/install.html
            "flashinfer-python",
            "sglang[all]>=0.5.1.post1",
            extra_options="--find-links https://flashinfer.ai/whl/cu126/torch2.6/flashinfer-python/",
            extra_index_url="https://flashinfer.ai/whl/cu126/torch2.6/",
        )
        .apt_install("libnuma-dev")
        .env(
            {
                # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
                "TORCH_CUDA_ARCH_LIST": "7.5 8.0 8.6 8.9 9.0 10.0",
            }
        )
    )


# img = img.pip_install(
#    f"achatbot==0.0.24.post29",
#    extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
# )

# ----------------------- app -------------------------------
app = modal.App("fastapi_webrtc_translate_bot")

HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)
TORCH_CACHE_DIR = "/root/.cache/torch"
torch_cache_vol = modal.Volume.from_name("torch_cache", create_if_missing=True)


# 128 MiB of memory and 0.125 CPU cores by default container runtime
@app.cls(
    image=img,
    gpu=os.getenv("IMAGE_GPU", None),
    secrets=[modal.Secret.from_name("achatbot")],
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_dir,
        TORCH_CACHE_DIR: torch_cache_vol,
    },
    cpu=2.0,
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
    # allow_concurrent_inputs=int(os.getenv("IMAGE_CONCURRENT_CN", "1")),
)
@modal.concurrent(max_inputs=int(os.getenv("IMAGE_CONCURRENT_CN", "1")))  # inputs per container
class Srv:
    @modal.enter()
    def enter(self):
        # run container runtime to enter when container is starting
        import subprocess
        import torch

        subprocess.run("nvidia-smi --version", shell=True)
        gpu_prop = None
        if torch.cuda.is_available():
            gpu_prop = torch.cuda.get_device_properties("cuda:0")
            print(gpu_prop)
            torch.multiprocessing.set_start_method("spawn", force=True)
        else:
            print("CUDA is not available.")

        # todo: init model to load, now use api to load model to run bot with config

    @modal.asgi_app()
    def app(self):
        from achatbot.cmd.http.server.fastapi_daily_bot_serve import app as fastapi_app

        return fastapi_app


"""
IMAGE_GPU=L4 LLM_TAG=llm_ctranslate2_generator \
    ACHATBOT_VERSION=0.0.24 \
    modal serve src/fastapi_webrtc_translate_bot_serve.py

IMAGE_GPU=L4 LLM_TAG=llm_vllm_generator \
    ACHATBOT_VERSION=0.0.24 \
    modal serve src/fastapi_webrtc_translate_bot_serve.py

IMAGE_GPU=L4 LLM_TAG=llm_sglang_generator \
    ACHATBOT_VERSION=0.0.24 \
    modal serve src/fastapi_ws_translate_bot_serve.py
"""
