import modal
import os

achatbot_version = os.getenv("ACHATBOT_VERSION", "0.0.18")
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
            "lite_avatar_gpu,"
            "queue"
            f"]=={achatbot_version}",
        ],
        extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
    )
    .pip_install(
        # fix Future exception was never retrieve, when connect timeout ( Connection reset by peer, retry)
        "aiohttp==3.10.11",
        "numpy==1.26.4",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "ACHATBOT_PKG": "1",
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "info"),
            "IMAGE_NAME": os.getenv("IMAGE_NAME", "default"),
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


# ----------------------- app -------------------------------
app = modal.App("fastapi_webrtc_avatar_bot")

HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
RESOURCES_DIR = "/root/.achatbot/resources"
resources_vol = modal.Volume.from_name("resources", create_if_missing=True)


# 128 MiB of memory and 0.125 CPU cores by default container runtime
@app.cls(
    image=image,
    gpu=os.getenv("IMAGE_GPU", None),
    secrets=[modal.Secret.from_name("achatbot")],
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        RESOURCES_DIR: resources_vol,
    },
    cpu=2.0,
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
    max_inputs=int(os.getenv("IMAGE_CONCURRENT_CN", "1")),
)
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
