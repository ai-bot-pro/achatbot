import modal
import os


class ContainerRuntimeConfig:
    images = {
        "default": (
            modal.Image.debian_slim(python_version="3.11")
            .apt_install("git", "git-lfs", "ffmpeg")
            .pip_install(
                [
                    "achatbot["
                    "fastapi_bot_server,"
                    "livekit,livekit-api,daily,agora,"
                    "silero_vad_analyzer,daily_langchain_rag_bot,"
                    "sense_voice_asr,deepgram_asr_processor,"
                    "openai_llm_processor,google_llm_processor,litellm_processor,"
                    "tts_edge,"
                    "deep_translator,together_ai,"
                    "queue"
                    "]~=0.0.8.3",
                    "huggingface_hub[hf_transfer]==0.24.7",
                    "wget",
                ],
                extra_index_url="https://pypi.org/simple/",
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
                    "GOOGLE_LLM_MODEL": "gemini-1.5-flash-latest",
                    # tts module engine TAG,default tts_edge
                    "TTS_TAG": "tts_edge",
                }
            )
        ),
    }

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


# 128 MiB of memory and 0.125 CPU cores by default container runtime
@app.cls(
    image=ContainerRuntimeConfig.get_img(),
    secrets=[modal.Secret.from_name("achatbot")],
    cpu=2.0,
    scaledown_window=300,
    timeout=600,
    allow_concurrent_inputs=100,
)
class Srv:
    @modal.build()
    def setup(self):
        # https://huggingface.co/docs/huggingface_hub/guides/download
        import wget
        from huggingface_hub import snapshot_download
        from achatbot.common.types import MODELS_DIR, ASSETS_DIR

        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(ASSETS_DIR, exist_ok=True)

        print(f"start downloading assets to dir:{ASSETS_DIR}")
        storytelling_assets = [
            "https://raw.githubusercontent.com/ai-bot-pro/achatbot/refs/heads/main/src/cmd/bots/image/storytelling/assets/book1.png",
            "https://raw.githubusercontent.com/ai-bot-pro/achatbot/refs/heads/main/src/cmd/bots/image/storytelling/assets/book2.png",
            "https://raw.githubusercontent.com/ai-bot-pro/achatbot/refs/heads/main/src/cmd/bots/image/storytelling/assets/ding.wav",
            "https://raw.githubusercontent.com/ai-bot-pro/achatbot/refs/heads/main/src/cmd/bots/image/storytelling/assets/listening.wav",
            "https://raw.githubusercontent.com/ai-bot-pro/achatbot/refs/heads/main/src/cmd/bots/image/storytelling/assets/talking.wav",
        ]

        for url in storytelling_assets:
            wget.download(url, out=ASSETS_DIR)

        print(f"start downloading model to dir:{MODELS_DIR}")
        snapshot_download(
            repo_id="FunAudioLLM/SenseVoiceSmall",
            repo_type="model",
            allow_patterns="*",
            local_dir=os.path.join(MODELS_DIR, "FunAudioLLM/SenseVoiceSmall"),
            # local_dir_use_symlinks=False,
        )
        print("setup done")

    @modal.enter()
    def enter(self):
        print("start enter")

    @modal.asgi_app()
    def app(self):
        from achatbot.cmd.http.server.fastapi_daily_bot_serve import app

        return app


if __name__ == "__main__":
    pass
