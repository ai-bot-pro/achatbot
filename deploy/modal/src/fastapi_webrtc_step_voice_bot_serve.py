import modal
import os


class ContainerRuntimeConfig:
    images = {
        "default": (
            modal.Image.debian_slim(python_version="3.11")
            .apt_install("git", "git-lfs", "ffmpeg", "sox")
            .pip_install(
                [
                    "achatbot["
                    "fastapi_bot_server,"
                    "livekit,livekit-api,daily,agora,"
                    "silero_vad_analyzer,"
                    "sense_voice_asr,deepgram_asr_processor,"
                    "tts_edge,"
                    "queue"
                    "]~=0.0.8.13",
                    "huggingface_hub[hf_transfer]==0.24.7",
                ],
                extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
            )
            .run_commands(
                f"pip install achatbot[step_voice_processor]==0.0.8.13 --extra-index-url {os.getenv('EXTRA_INDEX_URL', 'https://pypi.org/simple/')}"
            )
            .env(
                {
                    "HF_HUB_ENABLE_HF_TRANSFER": "1",
                    "ACHATBOT_PKG": "1",
                    # asr module engine TAG, default whisper_timestamped_asr
                    "ASR_TAG": "sense_voice_asr",
                    "ASR_LANG": "zn",
                    "ASR_MODEL_NAME_OR_PATH": "/root/.achatbot/models/FunAudioLLM/SenseVoiceSmall",
                    "LOG_LEVEL": os.getenv("LOG_LEVEL", "info"),
                    "IMAGE_NAME": os.getenv("IMAGE_NAME", "default"),
                    "ACHATBOT_TASK_DONE_TIMEOUT": os.getenv("ACHATBOT_TASK_DONE_TIMEOUT", "1200"),
                    "STEP_TTS_ASSETS_DIR": "/root/.achatbot/assets/",
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
        app_name = "fastapi_webrtc_step_voice_bot"
        if image_name != "default":
            app_name = f"fastapi_webrtc_step_voice_{image_name}_bot"
        print(f"app_name:{app_name}")
        return app_name

    @staticmethod
    def get_gpu():
        # T4, L4, A10G, A100, H100
        gpu = os.getenv("IMAGE_GPU", "T4")
        print(f"image_gpu:{gpu}")
        return gpu

    @staticmethod
    def get_allow_concurrent_inputs():
        # T4, L4, A10G, A100, H100
        concurrent_cn = int(os.getenv("IMAGE_CONCURRENT_CN", "1"))
        print(f"image_concurrent_cn:{concurrent_cn}")
        return concurrent_cn


img = ContainerRuntimeConfig.get_img()
with img.imports():
    import logging
    import os

    from achatbot.common.logger import Logger

MODEL_DIR = "/root/.achatbot/models"
model_dir = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assert_dir = modal.Volume.from_name("assets", create_if_missing=True)

# ----------------------- app -------------------------------
app = modal.App("fastapi_webrtc_step_voice_bot")

# volume = modal.Volume.from_name("bot_config", create_if_missing=True)

image = ContainerRuntimeConfig.get_img()
gpu = ContainerRuntimeConfig.get_gpu()
allow_concurrent_inputs = ContainerRuntimeConfig.get_allow_concurrent_inputs()


# 128 MiB of memory and 0.125 CPU cores by default container runtime
@app.cls(
    image=image,
    volumes={MODEL_DIR: model_dir, ASSETS_DIR: assert_dir},
    gpu=gpu,
    secrets=[modal.Secret.from_name("achatbot")],
    cpu=2.0,
    scaledown_window=300,
    timeout=600,
    allow_concurrent_inputs=allow_concurrent_inputs,
)
class Srv:
    # run download_models.py to download models to volume
    # @modal.build()
    # def setup(self):
    #     Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)
    #     # https://huggingface.co/docs/huggingface_hub/guides/download
    #     from huggingface_hub import snapshot_download
    #     from achatbot.common.types import MODELS_DIR

    #     os.makedirs(MODELS_DIR, exist_ok=True)
    #     logging.info(f"start downloading model to dir:{MODELS_DIR}")

    #     # asr model repo
    #     if "sense_voice_asr" in os.getenv("ASR_TAG", "sense_voice_asr"):
    #         local_dir = os.path.join(MODELS_DIR, "FunAudioLLM/SenseVoiceSmall")
    #         snapshot_download(
    #             repo_id="FunAudioLLM/SenseVoiceSmall",
    #             repo_type="model",
    #             allow_patterns="*",
    #             local_dir=local_dir,
    #         )
    #         logging.info(f"sense_voice_asr model to dir:{local_dir} done")

    #     for repo_id in [
    #         "stepfun-ai/Step-Audio-Tokenizer",
    #         "stepfun-ai/Step-Audio-TTS-3B",
    #         "stepfun-ai/Step-Audio-Chat",
    #     ]:
    #         logging.info(f"{repo_id} model to dir:{MODEL_DIR}")
    #         snapshot_download(
    #             repo_id=repo_id,
    #             repo_type="model",
    #             allow_patterns="*",
    #             local_dir=os.path.join(MODEL_DIR, repo_id),
    #         )
    #         logging.info(f"{repo_id} model to dir:{MODEL_DIR} done")

    #     print("download model done")

    @modal.enter()
    def enter(self):
        print("enter done")
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(ASSETS_DIR, exist_ok=True)
        # modal.Volume.reload()
        pass

    @modal.asgi_app()
    def app(self):
        from achatbot.cmd.http.server.fastapi_daily_bot_serve import app as fastapi_app

        return fastapi_app
