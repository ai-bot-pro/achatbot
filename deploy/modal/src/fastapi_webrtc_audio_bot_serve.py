import modal
import os

achatbot_version = os.getenv("ACHATBOT_VERSION", "0.0.15")


image = (
    # modal.Image.debian_slim(python_version="3.11")
    modal.Image.from_registry("node:22-slim", add_python="3.10")
    .apt_install("git", "git-lfs", "ffmpeg")
    .pip_install(
        [
            "achatbot["
            "fastapi_bot_server,"
            "livekit,livekit-api,daily,agora,"
            "silero_vad_analyzer,daily_langchain_rag_bot,"
            "sense_voice_asr,deepgram_asr_processor,"
            "openai_llm_processor,google_llm_processor,litellm_processor,"
            "deep_translator,together_ai,"
            "mcp,"
            "tts_edge,"
            "queue"
            f"]=={achatbot_version}",
        ],
        extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
    )
    .pip_install(
        "aiohttp==3.10.11"
    )  # fix Future exception was never retrieve, when connect timeout ( Connection reset by peer, retry)
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


class ContainerRuntimeConfig:
    images = {
        # default don't preinstall tools, when setup bot to install tools
        "default": image,
        # preinstall nasa mcp server tool
        "nasa": image.run_commands("npx -y @programcomputer/nasa-mcp-server@latest"),
        # preinstall amap mcp server tool when have amap api key
        "travel": image.env(
            {
                "AMAP_MAPS_API_KEY": os.getenv("AMAP_MAPS_API_KEY"),
            }
        ).run_commands("AMAP_MAPS_API_KEY=$AMAP_MAPS_API_KEY npx -y @amap/amap-maps-mcp-server")
        if os.getenv("AMAP_MAPS_API_KEY")
        else image,
    }

    @staticmethod
    def get_allow_concurrent_inputs():
        concurrent_cn = int(os.getenv("IMAGE_CONCURRENT_CN", "1"))
        print(f"image_concurrent_cn:{concurrent_cn}")
        return concurrent_cn

    @staticmethod
    def get_img(image_name: str = None):
        image_name = image_name or os.getenv("IMAGE_NAME", "default")
        if image_name not in ContainerRuntimeConfig.images:
            raise Exception(f"image name {image_name} not found")
        print(f"use image:{image_name}")
        return ContainerRuntimeConfig.images[image_name]

    @staticmethod
    def get_app_name(image_name: str = None):
        image_name = image_name or os.getenv("IMAGE_NAME", "default")
        if image_name != "default":
            return f"{image_name}_fastapi_webrtc_audio_bot"
        return "fastapi_webrtc_audio_bot"


# ----------------------- app -------------------------------
app = modal.App(ContainerRuntimeConfig.get_app_name())

ASSETS_DIR = "/root/.achatbot/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)
HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
TORCH_CACHE_DIR = "/root/.cache/torch"
torch_cache_vol = modal.Volume.from_name("torch_cache", create_if_missing=True)


# 128 MiB of memory and 0.125 CPU cores by default container runtime
@app.cls(
    image=ContainerRuntimeConfig.get_img(),
    secrets=[modal.Secret.from_name("achatbot")],
    volumes={ASSETS_DIR: assets_dir, HF_MODEL_DIR: hf_model_vol, TORCH_CACHE_DIR: torch_cache_vol},
    cpu=2.0,
    timeout=1200,  # default 300s
    scaledown_window=1200,
)
@modal.concurrent(max_inputs=int(os.getenv("IMAGE_CONCURRENT_CN", "1")))  # inputs per container
class Srv:
    @modal.enter()
    def enter(self):
        import subprocess

        subprocess.run("which npx", shell=True)
        subprocess.run("npx --version", shell=True)

    @modal.asgi_app()
    def app(self):
        from achatbot.cmd.http.server.fastapi_daily_bot_serve import app

        return app


if __name__ == "__main__":
    pass
