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
                    "livekit,livekit-api,daily,"
                    "silero_vad_analyzer,"
                    "sense_voice_asr,"
                    "openai_llm_processor,google_llm_processor,litellm_processor,"
                    "tts_edge"
                    "]~=0.0.7.10",
                    "huggingface_hub[hf_transfer]==0.24.7",
                ],
                extra_index_url="https://pypi.org/simple/")
            .env({
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "ACHATBOT_PKG": "1",
                "LOG_LEVEL": os.getenv("LOG_LEVEL", "info"),
            })
        ),
    }

    @staticmethod
    def get_img(image_name: str = None):
        image_name = image_name or os.getenv("IMAGE_NAME", "default")
        if image_name not in ContainerRuntimeConfig.images:
            raise Exception(f"image name {image_name} not found")
        print(f"use image:{image_name}")
        return ContainerRuntimeConfig.images[image_name]


# ----------------------- app -------------------------------
app = modal.App("fastapi_webrtc_audio_bot")


# 128 MiB of memory and 0.125 CPU cores by default container runtime
@app.cls(
    image=ContainerRuntimeConfig.get_img(),
    secrets=[modal.Secret.from_name("achatbot")],
    cpu=2.0,
    container_idle_timeout=300,
    timeout=600,
    allow_concurrent_inputs=100,
)
class Srv:
    @modal.build()
    def download_model(self):
        # https://huggingface.co/docs/huggingface_hub/guides/download
        from huggingface_hub import snapshot_download
        from achatbot.common.types import MODELS_DIR
        os.makedirs(MODELS_DIR, exist_ok=True)
        print(f"start downloading model to dir:{MODELS_DIR}")

        snapshot_download(
            repo_id="FunAudioLLM/SenseVoiceSmall",
            repo_type="model",
            allow_patterns="*",
            local_dir=os.path.join(MODELS_DIR, "FunAudioLLM/SenseVoiceSmall"),
            # local_dir_use_symlinks=False,
        )
        print("download model done")

    @modal.enter()
    def enter(self):
        print("start enter")

    @modal.asgi_app()
    def app(self):
        from achatbot.cmd.http.server.fastapi_daily_bot_serve import app
        return app


if __name__ == "__main__":
    pass
