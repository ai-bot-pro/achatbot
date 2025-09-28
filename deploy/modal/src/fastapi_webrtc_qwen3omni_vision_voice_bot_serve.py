import modal
import os

achatbot_version = os.getenv("ACHATBOT_VERSION", "0.0.26.post1")
# fastapi_webrtc_bots | fastapi_webrtc_single_bot server
SERVER_TAG = os.getenv("SERVER_TAG", "fastapi_webrtc_bots")
IMAGE_GPU = os.getenv("IMAGE_GPU", "A100-80GB")
img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "clang", "cmake")
    .pip_install("wheel", "openai", "qwen-omni-utils[decord]")
    .pip_install(
        [
            "achatbot["
            "fastapi_bot_server,"
            "livekit,livekit-api,daily,agora,"
            "silero_vad_analyzer,asr_processor,"
            "queue"
            f"]=={achatbot_version}",
        ],
        extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
    )
    .run_commands("pip install git+https://github.com/huggingface/transformers")
    .pip_install(
        "accelerate",
        "torch==2.7.0",
        "torchaudio==2.7.0",
        "torchvision==0.22.0",
        "soundfile==0.13.0",
        "librosa==0.11.0",
    )
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    .env(
        {
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "info"),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "ACHATBOT_PKG": "1",
            "SERVER_TAG": SERVER_TAG,
            "CONFIG_FILE": os.getenv(
                "CONFIG_FILE",
                "/root/.achatbot/config/bots/livekit_qwen3omni_vision_voice_bot.json",
            ),
            # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
        }
    )
)

# img = img.pip_install(
#    f"achatbot==0.0.26.dev20",
#    extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://test.pypi.org/simple/"),
# )

# ----------------------- app -------------------------------
app = modal.App("qwen3omni_bot")

HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)
CONFIG_DIR = "/root/.achatbot/config"
config_vol = modal.Volume.from_name("config", create_if_missing=True)
RECORDS_DIR = "/root/.achatbot/records"
records_vol = modal.Volume.from_name("records", create_if_missing=True)
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
        CONFIG_DIR: config_vol,
        RECORDS_DIR: records_vol,
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

    @modal.asgi_app()
    def app(self):
        SERVER_TAG = os.getenv("SERVER_TAG", "fastapi_webrtc_bots")
        if SERVER_TAG == "fastapi_webrtc_single_bot":
            from achatbot.cmd.http.server.fastapi_room_bot_serve import app as fastapi_app

            print("run fastapi_room_bot_serve(single bot)")
        else:
            from achatbot.cmd.http.server.fastapi_daily_bot_serve import app as fastapi_app

            print("run fastapi_daily_bot_serve(multi bots)")

        return fastapi_app


"""
# 0. download models and assets
modal run src/download_models.py --repo-ids "Qwen/Qwen3-Omni-30B-A3B-Instruct"

# 1. run webrtc room http bots server

IMAGE_GPU=A100-80GB SERVER_TAG=fastapi_webrtc_bots \
    ACHATBOT_VERSION=0.0.26.post1 \
    modal serve src/fastapi_webrtc_qwen3omni_vision_voice_bot_serve.py

# 2. run webrtc room http signal bot server

modal volume create config
modal volume put config ./config/bots/livekit_qwen3omni_vision_voice_bot.json /bots/ -f


# run container with gpu
IMAGE_GPU=A100-80GB SERVER_TAG=fastapi_webrtc_bots \
    ACHATBOT_VERSION=0.0.26.post1 \
    CONFIG_FILE=/root/.achatbot/config/bots/livekit_qwen3omni_vision_voice_bot.json \
    modal serve src/fastapi_webrtc_qwen3omni_vision_voice_bot_serve.py

# cold start fastapi webrtc http server
curl -v -XGET "https://weedge--qwen3omni-bot-srv-app-dev.modal.run/health"

# run bot
curl -XPOST "https://weedge--qwen3omni-bot-srv-app-dev.modal.run/bot_join/chat-room/LivekitQwen3OmniVisionVoiceBot"


"""
