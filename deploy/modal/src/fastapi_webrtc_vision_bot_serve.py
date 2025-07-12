import modal
import os

achatbot_version = os.getenv("ACHATBOT_VERSION", "0.0.16")
secret = os.getenv("SECRET_NAME", "achatbot")

SERVE_TYPE = os.getenv("SERVE_TYPE", "room_bots")  # room_bot,room_bots
IMAGE_GPU = os.getenv("IMAGE_GPU", "A100")
IMAGE_NAME = os.getenv("IMAGE_NAME", "default")
FASTDEPLOY_VERSION = os.getenv("FASTDEPLOY_VERSION", "stable")  # stable, nightly
GPU_ARCHS = os.getenv("GPU_ARCHS", "80_90")  # 80_90, 86_89
CONFIG_FILE = os.getenv(
    "CONFIG_FILE",
    "/root/.achatbot/config/bots/dummy_bot.json",
)

vision_bot_img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "cmake", "ninja-build")
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
            "IMAGE_NAME": IMAGE_NAME,
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
                    "LLM_MODEL_NAME_OR_PATH": f"/root/.achatbot/models/{os.getenv('LLM_MODEL_NAME_OR_PATH', 'Qwen/Qwen2-VL-2B-Instruct')}",
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
                    "LLM_MODEL_NAME_OR_PATH": f"/root/.achatbot/models/{os.getenv('LLM_MODEL_NAME_OR_PATH', 'unsloth/Llama-3.2-11B-Vision-Instruct')}",
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
                    "LLM_MODEL_NAME_OR_PATH": f"/root/.achatbot/models/{os.getenv('LLM_MODEL_NAME_OR_PATH', 'deepseek-ai/Janus-Pro-1B')}",
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
                    "LLM_MODEL_NAME_OR_PATH": f"/root/.achatbot/models/{os.getenv('LLM_MODEL_NAME_OR_PATH', 'deepseek-ai/deepseek-vl2-tiny')}",
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
                    "LLM_MODEL_NAME_OR_PATH": f"/root/.achatbot/models/{os.getenv('LLM_MODEL_NAME_OR_PATH', 'openbmb/MiniCPM-o-2_6')}",
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
            .run_commands(
                "pip install git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview"
            )
            .env(
                {
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    "LLM_MODEL_NAME_OR_PATH": f"/root/.achatbot/models/{os.getenv('LLM_MODEL_NAME_OR_PATH', 'Qwen/Qwen2.5-Omni-7B')}",
                }
            )
        ),
        "fastvlm": (
            vision_bot_img.pip_install(
                [
                    f"achatbot[llm_transformers_manual_vision_fastvlm]=={achatbot_version}",
                ],
                extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
            ).env(
                {
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    "LLM_MODEL_NAME_OR_PATH": f"/root/.achatbot/models/{os.getenv('LLM_MODEL_NAME_OR_PATH', 'llava-fastvithd_1.5b_stage3')}",
                    "MOBILE_CLIP_MODEL_CONFIG": "/root/.achatbot/models/mobileclip_l.json",
                }
            )
        ),
        "smolvlm": (
            vision_bot_img.pip_install(
                [
                    f"achatbot[llm_transformers_manual_vision_smolvlm]=={achatbot_version}",
                ],
                extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
            ).env(
                {
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    "LLM_MODEL_NAME_OR_PATH": f"/root/.achatbot/models/{os.getenv('LLM_MODEL_NAME_OR_PATH', 'HuggingFaceTB/SmolVLM2-2.2B-Instruct')}",
                }
            )
        ),
        "gemma3": (
            vision_bot_img.pip_install(
                [
                    f"achatbot[llm_transformers_manual_vision_gemma]=={achatbot_version}",
                ],
                extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
            ).env(
                {
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    "LLM_MODEL_NAME_OR_PATH": f"/root/.achatbot/models/{os.getenv('LLM_MODEL_NAME_OR_PATH', 'google/gemma-3-4b-it')}",
                }
            )
        ),
        "phi4": (
            vision_bot_img.pip_install(
                [
                    f"achatbot[llm_transformers_manual_vision_speech_phi]=={achatbot_version}",
                ],
                extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
            ).env(
                {
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    "LLM_MODEL_NAME_OR_PATH": f"/root/.achatbot/models/{os.getenv('LLM_MODEL_NAME_OR_PATH', 'microsoft/Phi-4-multimodal-instruct')}",
                }
            )
        ),
        "mimo": (
            vision_bot_img.pip_install(
                [
                    f"achatbot[llm_transformers_manual_vision_mimo]=={achatbot_version}",
                ],
                extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
            ).env(
                {
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    "LLM_MODEL_NAME_OR_PATH": f"/root/.achatbot/models/{os.getenv('LLM_MODEL_NAME_OR_PATH', 'XiaomiMiMo/MiMo-VL-7B-RL')}",
                }
            )
        ),
        "keye": (
            vision_bot_img.pip_install(
                [
                    f"achatbot[llm_transformers_manual_vision_keye]=={achatbot_version}",
                ],
                extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
            )
            .run_commands(
                "pip install git+https://github.com/huggingface/transformers@17b3c96c00cd8421bff85282aec32422bdfebd31"
            )
            .pip_install("accelerate")
            .env(
                {
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    "LLM_MODEL_NAME_OR_PATH": f"/root/.achatbot/models/{os.getenv('LLM_MODEL_NAME_OR_PATH', 'Kwai-Keye/Keye-VL-8B-Preview')}",
                }
            )
        ),
        "glm4v": (
            vision_bot_img.pip_install(
                [
                    f"achatbot[llm_transformers_manual_vision_glm4v]=={achatbot_version}",
                ],
                extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
            )
            .run_commands(
                "pip install git+https://github.com/huggingface/transformers@17b3c96c00cd8421bff85282aec32422bdfebd31"
            )
            .pip_install("accelerate")
            .env(
                {
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    "LLM_MODEL_NAME_OR_PATH": f"/root/.achatbot/models/{os.getenv('LLM_MODEL_NAME_OR_PATH', 'THUDM/GLM-4.1V-9B-Thinking')}",
                }
            )
        ),
        "ernie4v": (
            vision_bot_img.pip_install(
                [
                    f"achatbot[llm_transformers_manual_vision]=={achatbot_version}",
                ],
                extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
            )
        ),
        "fastdeploy_ernie4v": (
            vision_bot_img.pip_install(
                "paddlepaddle-gpu==3.1.0",
                index_url=" https://www.paddlepaddle.org.cn/packages/stable/cu126/",
            ).run_commands(
                f"python -m pip install fastdeploy-gpu -i https://www.paddlepaddle.org.cn/packages/{FASTDEPLOY_VERSION}/fastdeploy-gpu-{GPU_ARCHS}/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
            )
        ),
    }

    @staticmethod
    def get_img(image_name: str = None):
        image_name = image_name or IMAGE_NAME
        if image_name not in ContainerRuntimeConfig.images:
            raise Exception(f"image name {image_name} not found")
        print(f"use image:{image_name}")
        return ContainerRuntimeConfig.images[image_name]

    @staticmethod
    def get_app_name(image_name: str = None):
        image_name = image_name or IMAGE_NAME
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


if IMAGE_NAME not in ["fastdeploy_ernie4v"]:
    img = ContainerRuntimeConfig.get_img().pip_install(
        "flash-attn==2.7.4.post1", extra_options="--no-build-isolation"
    )
else:
    img = ContainerRuntimeConfig.get_img()


if SERVE_TYPE == "room_bot":
    img = img.env(
        {
            "CONFIG_FILE": CONFIG_FILE,
        }
    )

# img = img.pip_install(
#    f"achatbot==0.0.21",
#    extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
# )

HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)
TORCH_CACHE_DIR = "/root/.cache/torch"
torch_cache_vol = modal.Volume.from_name("torch_cache", create_if_missing=True)
CONFIG_DIR = "/root/.achatbot/config"
config_vol = modal.Volume.from_name("config", create_if_missing=True)


# ----------------------- app -------------------------------
app = modal.App(ContainerRuntimeConfig.get_app_name())


# 128 MiB of memory and 0.125 CPU cores by default container runtime
@app.cls(
    image=img.env({"SERVE_TYPE": SERVE_TYPE}),
    gpu=ContainerRuntimeConfig.get_gpu(),
    secrets=[modal.Secret.from_name(secret)],
    retries=0,
    cpu=2.0,
    # allow_concurrent_inputs=ContainerRuntimeConfig.get_allow_concurrent_inputs(),
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_dir,
        TORCH_CACHE_DIR: torch_cache_vol,
        CONFIG_DIR: config_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=int(os.getenv("IMAGE_MAX_CONTAINERS", "1")),
)
@modal.concurrent(max_inputs=int(os.getenv("IMAGE_CONCURRENT_CN", "1")))  # inputs per container
class Srv:
    @modal.enter()
    def enter(self):
        # run container runtime to enter when container is starting
        import subprocess
        import torch

        from achatbot.common.logger import Logger

        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)

        subprocess.run("nvidia-smi --version", shell=True)
        gpu_prop = None
        if torch.cuda.is_available():
            gpu_prop = torch.cuda.get_device_properties("cuda:0")
            print(gpu_prop)
            IMAGE_NAME = os.getenv("IMAGE_NAME")
            if "fastdeploy" not in IMAGE_NAME:
                torch.multiprocessing.set_start_method("spawn", force=True)
        else:
            print("CUDA is not available.")

    @modal.asgi_app()
    def app(self):
        SERVE_TYPE = os.getenv("SERVE_TYPE")
        if SERVE_TYPE == "room_bot":
            from achatbot.cmd.http.server.fastapi_room_bot_serve import app
        else:
            from achatbot.cmd.http.server.fastapi_daily_bot_serve import app

        return app


"""

# run dummy bot to join room for test
EXTRA_INDEX_URL=https://pypi.org/simple/ \
    SERVE_TYPE=room_bot \
    ACHATBOT_VERSION=0.0.21 \
    IMAGE_NAME=fastdeploy_ernie4v IMAGE_CONCURRENT_CN=1 IMAGE_GPU=L40s \
    GPU_ARCHS=86_89 \
    modal serve src/fastapi_webrtc_vision_bot_serve.py

# run fastdeploy ernie4v bot to join room
EXTRA_INDEX_URL=https://pypi.org/simple/ \
    SERVE_TYPE=room_bot \
    CONFIG_FILE=/root/.achatbot/config/bots/daily_describe_fastdeploy_ernie4v_vision_bot.json \
    ACHATBOT_VERSION=0.0.21 \
    IMAGE_NAME=fastdeploy_ernie4v IMAGE_CONCURRENT_CN=1 IMAGE_GPU=L40s \
    GPU_ARCHS=86_89 \
    modal serve src/fastapi_webrtc_vision_bot_serve.py
"""
