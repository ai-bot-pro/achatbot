import modal
import os

achatbot_version = os.getenv("ACHATBOT_VERSION", "0.0.19")
avatar_tag = os.getenv("AVATAR_TAG", "lite_avatar_gpu")
image = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
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
            "openai_llm_processor,google_llm_processor,litellm_processor,"
            "deep_translator,together_ai,"
            "tts_edge,"
            f"{avatar_tag},"
            "queue"
            f"]=={achatbot_version}",
        ],
        extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
    )
    .pip_install(
        # fix Future exception was never retrieve, when connect timeout ( Connection reset by peer, retry)
        "aiohttp==3.10.11",
        "numpy==1.26.4",
        "torch==2.4.1",
        "torchvision==0.19.1",
        "torchaudio==2.4.1",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "ACHATBOT_PKG": "1",
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "info"),
            # asr module engine TAG, default whisper_timestamped_asr
            "ASR_TAG": "sense_voice_asr",
            "ASR_LANG": "zn",
            "ASR_MODEL_NAME_OR_PATH": "/root/.achatbot/models/FunAudioLLM/SenseVoiceSmall",
            # llm processor model, default:google gemini_flash_latest
            "GOOGLE_LLM_MODEL": "gemini-2.0-flash-lite",
            # tts module engine TAG,default tts_edge
            "TTS_TAG": "tts_edge",
        }
    )
)

if avatar_tag == "musetalk_avatar":
    image = (
        image.apt_install("clang")
        .run_commands(
            "which nvcc",
            "which clang++",
            "git clone https://github.com/open-mmlab/mmcv.git",
            "cd /mmcv && git checkout v2.1.0",
            "cd /mmcv && pip install -r requirements/optional.txt",
            "cd /mmcv && FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST='7.5 8.0 8.6 8.7 8.9 9.0' pip install -e . -v",
        )
        .run_commands(
            "pip install -q --no-cache-dir -U openmim",
            "mim install mmengine",
            "mim install 'mmdet==3.3.0'",
            "mim install 'mmpose==1.3.2'",
        )
    )

if avatar_tag == "lam_audio2expression_avatar":
    image = (
        image.pip_install("spleeter==2.4.2")
        .pip_install(
            "typing_extensions==4.14.0",
            "aiortc==1.13.0",
            "protobuf==5.29.4",
            "transformers==4.36.2",
        )
        .env(
            {
                "TRANSPORT": os.getenv("TRANSPORT", "webrtc_websocket_v2"),
                "CONFIG_FILE": os.getenv(
                    "CONFIG_FILE",
                    "/root/.achatbot/config/bots/small_webrtc_fastapi_websocket_avatar_echo_bot.json",
                ),
            }
        )
    )

# image = image.pip_install(
# f"achatbot==0.0.19.post5",
# extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
# )

# ----------------------- app -------------------------------
app = modal.App("fastapi_webrtc_avatar_bot")

HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
RESOURCES_DIR = "/root/.achatbot/resources"
resources_vol = modal.Volume.from_name("resources", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_vol = modal.Volume.from_name("assets", create_if_missing=True)
CONFIG_DIR = "/root/.achatbot/config"
config_vol = modal.Volume.from_name("config", create_if_missing=True)
TORCH_CACHE_DIR = "/root/.cache/torch"
torch_cache_vol = modal.Volume.from_name("torch_cache", create_if_missing=True)


# 128 MiB of memory and 0.125 CPU cores by default container runtime
@app.cls(
    image=image,
    gpu=os.getenv("IMAGE_GPU", None),
    secrets=[modal.Secret.from_name("achatbot")],
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        RESOURCES_DIR: resources_vol,
        ASSETS_DIR: assets_vol,
        CONFIG_DIR: config_vol,
        TORCH_CACHE_DIR: torch_cache_vol,
    },
    cpu=4.0,
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=int(os.getenv("IMAGE_MAX_CONTAINERS", "1")),
)
@modal.concurrent(max_inputs=int(os.getenv("IMAGE_CONCURRENT_CN", "10")))  # inputs per container
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
        transport = os.getenv("TRANSPORT", "daily")
        if transport == "webrtc_websocket":
            from achatbot.cmd.webrtc_websocket.fastapi_ws_signaling_bot_serve import (
                app as fastapi_app,
            )

            return fastapi_app
        elif transport == "webrtc_websocket_v2":
            from achatbot.cmd.webrtc_websocket.fastapi_ws_signaling_bot_serve_v2 import (
                app as fastapi_app,
            )

            return fastapi_app
        else:
            from achatbot.cmd.http.server.fastapi_daily_bot_serve import app as fastapi_app

            return fastapi_app
