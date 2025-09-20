import os

import modal


achatbot_version = os.getenv("ACHATBOT_VERSION", "0.0.25.post2")
app = modal.App("step-audio2-voice-bot")
IMAGE_GPU = os.getenv("IMAGE_GPU", "L4")
img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvcr.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg")
    .pip_install(
        [
            "achatbot["
            "fastapi_bot_server,"
            "livekit,livekit-api,daily,agora,"
            "silero_vad_analyzer,"
            "sense_voice_asr,deepgram_asr_processor,"
            "tts_edge,"
            "queue"
            f"]~={achatbot_version}",
        ],
        extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
    )
    .pip_install(
        "transformers==4.49.0",
        "torchaudio",
        "librosa",
        "onnxruntime",
        "s3tokenizer",
        "diffusers",
        "hyperpyyaml",
        "huggingface_hub",
    )
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "ACHATBOT_PKG": "1",
            "CONFIG_FILE": os.getenv(
                "CONFIG_FILE",
                "/root/.achatbot/config/bots/fastapi_websocket_step_audio2_stst_bot.json",
            ),
        }
    )
)

# img = img.pip_install(
#   f"achatbot==0.0.25.dev122",
#   extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://test.pypi.org/simple/"),
# )


HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_vol = modal.Volume.from_name("assets", create_if_missing=True)
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
        ASSETS_DIR: assets_vol,
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
        from achatbot.cmd.websocket.server.fastapi_ws_bot_serve import app as fastapi_app

        return fastapi_app


"""

# run webrtc room http signal bot server

modal volume create config

modal volume put config ./config/bots/fastapi_websocket_step_audio2_s2st_bot.json /bots/ -f

# run container with gpu
IMAGE_GPU=L4 ACHATBOT_VERSION=0.0.25.post2 \
    CONFIG_FILE=/root/.achatbot/config/bots/fastapi_websocket_step_audio2_s2st_bot.json \
    modal serve src/fastapi_ws_step2_voice_bot_serve.py

# cold start fastapi websocket server
curl -v -XGET "https://weedge--step-audio2-voice-bot-srv-app-dev.modal.run/health"

# run websocket ui
cd ui/websocket && python -m http.server
# - access http://localhost:8000/translation   
# - change url to wss://weedge--fastapi-ws-translate-bot-srv-app-dev.modal.run
# - click `Start Audio` to speech translation with Translation bot
"""

"""
# 1. run step-audio2 vllm docker server
IMAGE_GPU=L4 modal serve src/llm/vllm/step_audio2.py
# cold start step-audio2 vllm docker server
curl -v -XGET "https://weedge--vllm-step-audio2-serve-dev.modal.run/health"

# 2. run webrtc room http signal bot server with vllm cli

modal volume create config
modal volume put config ./config/bots/fastapi_websocket_step_audio2_vllm_cli_s2st_bot.json /bots/ -f

## run container with gpu
IMAGE_GPU=T4 ACHATBOT_VERSION=0.0.25.post3 \
    CONFIG_FILE=/root/.achatbot/config/bots/fastapi_websocket_step_audio2_vllm_cli_s2st_bot.json \
    modal serve src/fastapi_ws_step2_voice_bot_serve.py

## cold start fastapi websocket server
curl -v -XGET "https://weedge--step-audio2-voice-bot-srv-app-dev.modal.run/health"

# 3. run websocket ui
cd ui/websocket && python -m http.server
# - access http://localhost:8000/translation   
# - change url to wss://weedge--step-audio2-voice-bot-srv-app-dev.modal.run
# - click `Start Audio` to speech translation with Translation bot
"""
