import modal
import os

achatbot_version = os.getenv("ACHATBOT_VERSION", "0.0.9.post10")

vision_bot_img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "cmake")
    .pip_install("wheel")
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
            f"]=={achatbot_version}",
        ],
        extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
    )
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    .env(
        {
            "ACHATBOT_PKG": "1",
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "info"),
            # asr module engine TAG, default whisper_timestamped_asr
            "ASR_TAG": "sense_voice_asr",
            "ASR_LANG": "zn",
            # "ASR_MODEL_NAME_OR_PATH": "/root/.achatbot/models/FunAudioLLM/SenseVoiceSmall",
            # llm processor model, default:google gemini_flash_latest
            "GOOGLE_LLM_MODEL": "gemini-2.0-flash",
            # tts module engine TAG,default tts_edge
            "TTS_TAG": "tts_edge",
        }
    )
)

# NOTE:
# LLM_MODEL_NAME_OR_PATH now is not used in the image
# use download_model.py to download model to models volume mount to /root/.achatbot/models


class ContainerRuntimeConfig:
    images = {
        "default": vision_bot_img,
        "qwen": (
            vision_bot_img.pip_install(
                [
                    f"achatbot[llm_transformers_manual_vision_qwen]=={achatbot_version}",
                ],
                extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
            ).env(
                {
                    "LLM_MODEL_NAME_OR_PATH": f'/root/.achatbot/models/{os.getenv("LLM_MODEL_NAME_OR_PATH", "Qwen/Qwen2-VL-2B-Instruct")}',
                }
            )
        ),
        "llama": (
            vision_bot_img.pip_install(
                [
                    f"achatbot[llm_transformers_manual_vision_llama]=={achatbot_version}",
                ],
                extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
            ).env(
                {
                    "LLM_MODEL_NAME_OR_PATH": f'/root/.achatbot/models/{os.getenv("LLM_MODEL_NAME_OR_PATH", "unsloth/Llama-3.2-11B-Vision-Instruct")}',
                }
            )
        ),
        "janus": (
            vision_bot_img.pip_install(
                [
                    f"achatbot[llm_transformers_manual_vision_img_janus]=={achatbot_version}",
                ],
                extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
            ).env(
                {
                    "LLM_MODEL_NAME_OR_PATH": f'/root/.achatbot/models/{os.getenv("LLM_MODEL_NAME_OR_PATH", "deepseek-ai/Janus-Pro-1B")}',
                }
            )
        ),
        "deepseekvl2": (
            vision_bot_img.pip_install(
                [
                    f"achatbot[llm_transformers_manual_vision_deepseekvl2]=={achatbot_version}",
                ],
                extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
            ).env(
                {
                    "LLM_MODEL_NAME_OR_PATH": f'/root/.achatbot/models/{os.getenv("LLM_MODEL_NAME_OR_PATH", "deepseek-ai/deepseek-vl2-tiny")}',
                }
            )
        ),
        "minicpmo": (
            vision_bot_img.pip_install(
                [
                    f"achatbot[llm_transformers_manual_vision_voice_minicpmo]=={achatbot_version}",
                ],
                extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
            ).env(
                {
                    "LLM_MODEL_NAME_OR_PATH": f'/root/.achatbot/models/{os.getenv("LLM_MODEL_NAME_OR_PATH", "openbmb/MiniCPM-o-2_6")}',
                }
            )
        ),
        "kimi": (
            vision_bot_img.pip_install(
                [
                    f"achatbot[llm_transformers_manual_vision_kimi]=={achatbot_version}",
                ],
                extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
            ).env(
                {
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                }
            )
        ),
        "qwen2_5omni": (
            vision_bot_img.pip_install(
                [
                    f"achatbot[llm_transformers_manual_vision_voice_qwen]=={achatbot_version}",
                ],
                extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
            )
            .run_commands("pip install git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview")
            .env(
                {
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    "LLM_MODEL_NAME_OR_PATH": f'/root/.achatbot/models/{os.getenv("LLM_MODEL_NAME_OR_PATH", "Qwen/Qwen2.5-Omni-7B")}',
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
        app_name = "fastapi_webrtc_vision_bot"
        if image_name != "default":
            app_name = f"fastapi_webrtc_vision_{image_name}_bot"
        print(f"app_name:{app_name}")
        return app_name

    @staticmethod
    def get_gpu():
        # https://modal.com/docs/reference/modal.gpu
        # T4, L4, A10G, L40S, A100, A100-80GB, H100
        gpu = os.getenv("IMAGE_GPU", None)
        return gpu

    @staticmethod
    def get_allow_concurrent_inputs():
        concurrent_cn = int(os.getenv("IMAGE_CONCURRENT_CN", "1"))
        print(f"image_concurrent_cn:{concurrent_cn}")
        return concurrent_cn


HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)


# ----------------------- app -------------------------------
app = modal.App(ContainerRuntimeConfig.get_app_name())


# 128 MiB of memory and 0.125 CPU cores by default container runtime
@app.cls(
    image=ContainerRuntimeConfig.get_img(),
    gpu=ContainerRuntimeConfig.get_gpu(),
    secrets=[modal.Secret.from_name("achatbot")],
    cpu=2.0,
    allow_concurrent_inputs=ContainerRuntimeConfig.get_allow_concurrent_inputs(),
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_dir,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
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

    @modal.asgi_app()
    def app(self):
        from achatbot.cmd.http.server.fastapi_daily_bot_serve import app

        return app


if __name__ == "__main__":
    pass
