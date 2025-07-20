import os
import sys
import asyncio

import modal

CONFIG_PATH = os.getenv("CONFIG_PATH", "config/chat_with_minicpm_gs.yaml")

image = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    # https://hub.docker.com/r/pytorch/pytorch/tags
    modal.Image.from_registry(
        "pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel",
        add_python="3.11",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "clang")
    .env(
        {
            "USE_GPTQ_CKPT": os.getenv("USE_GPTQ_CKPT", ""),
            # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
            # nvcc --list-gpu-arch
            "TORCH_CUDA_ARCH_LIST": "7.5;8.0;8.6+PTX;8.9+PTX;9.0",  # for auto-gptq install
        }
    )
    .run_commands(
        "which nvcc",
        "nvcc --version",
        "pip install --no-build-isolation achatbot[flash-attn]",
    )
    .run_commands(
        [
            "if [ $USE_GPTQ_CKPT ];then"
            + " git clone https://github.com/OpenBMB/AutoGPTQ.git -b minicpmo"
            + " && cd AutoGPTQ"
            + " && pip install -vvv --no-build-isolation -e . "
            + ";fi",
        ]
    )
    .pip_install("uv")
    .run_commands(
        "pip install chumpy==0.70 --no-build-isolation",
        "pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html",
    )
    .run_commands(
        "which python",
        "python --version",
        "git lfs install --skip-smudge",
        "git clone https://github.com/weedge/OpenAvatarChat.git -b feat/achatbot",
        "cd /OpenAvatarChat && git submodule foreach git lfs pull",
        "cd /OpenAvatarChat && git lfs install --force",
        "cd /OpenAvatarChat && git checkout 468ae5d9d6b1481b68ee94d3decdc8efcafbd105",
        "cd /OpenAvatarChat && git submodule update --init --recursive;git lfs logs last",
    )
    .run_commands(
        "cd /OpenAvatarChat && git pull origin feat/achatbot && git checkout 136725008748e69c47d3444f0b02bc658269bd26",
        f"cd /OpenAvatarChat && python install.py --uv --config {CONFIG_PATH}",
        "rm -rf /OpenAvatarChat/models",
        "rm -rf /OpenAvatarChat/resource",
    )
    .env(
        {
            "CONFIG_PATH": CONFIG_PATH,
        }
    )
    .pip_install("twilio", "pydantic==2.11.7")
)


# ----------------------- app -------------------------------
app = modal.App("open_avatar_chat_bot")


HF_MODEL_DIR = "/OpenAvatarChat/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
RESOURCES_DIR = "/OpenAvatarChat/resource"
resources_vol = modal.Volume.from_name("resources", create_if_missing=True)
TORCH_CACHE_DIR = "/root/.cache/torch"
torch_cache_vol = modal.Volume.from_name("torch_cache", create_if_missing=True)

"""
# download model weights
modal run src/download_models.py --repo-ids "FunAudioLLM/SenseVoiceSmall"
modal run src/download_models.py --repo-ids "openbmb/MiniCPM-o-2_6,openbmb/MiniCPM-o-2_6-int4" 
modal run src/download_models.py --repo-ids "facebook/wav2vec2-base-960h"
modal run src/download_models.py::download_ckpts --ckpt-urls "https://huggingface.co/3DAIGC/LAM_audio2exp/resolve/main/LAM_audio2exp_streaming.tar"

IMAGE_GPU=T4 CONFIG_PATH=config/echo_with_gs.yaml modal serve src/avatar/open_avatar_chat.py
IMAGE_GPU=T4 CONFIG_PATH=config/chat_with_gs.yaml modal serve src/avatar/open_avatar_chat.py

IMAGE_GPU=L4 CONFIG_PATH=config/chat_with_minicpm_gs.yaml modal serve src/avatar/open_avatar_chat.py
IMAGE_GPU=T4 USE_GPTQ_CKPT=1 CONFIG_PATH=config/chat_with_minicpm_int4_gs.yaml modal serve src/avatar/open_avatar_chat.py

"""


# RTC_CONFIG = {"iceServers": [{"url": "stun:stun.l.google.com:19302"}]}


# 128 MiB of memory and 0.125 CPU cores by default container runtime
@app.function(
    image=image,
    gpu=os.getenv("IMAGE_GPU", "T4"),
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        RESOURCES_DIR: resources_vol,
        TORCH_CACHE_DIR: torch_cache_vol,
    },
    secrets=[modal.Secret.from_name("achatbot")],
    cpu=4.0,
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
    # NOTE: don't use max_inputs in function params
    # max_inputs=int(os.getenv("IMAGE_CONCURRENT_CN", "1")),
)
@modal.concurrent(max_inputs=int(os.getenv("IMAGE_CONCURRENT_CN", "10")))  # inputs per container
@modal.asgi_app()  # ASGI on Modal
def ui():
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

    return init_app()


def init_app():
    import gradio as gr
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import RedirectResponse
    from gradio.routes import mount_gradio_app  # connects Gradio and FastAPI

    sys.path.insert(0, "/OpenAvatarChat/src")

    from engine_utils.directory_info import DirectoryInfo
    from service.service_utils.service_config_loader import load_configs
    from service.service_utils.logger_utils import config_loggers
    from chat_engine.chat_engine import ChatEngine

    project_dir = DirectoryInfo.get_project_dir()
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)

    logger_config, service_config, engine_config = load_configs(os.getenv("CONFIG_PATH"))

    config_loggers(logger_config)

    # set up demo
    app = FastAPI()

    @app.get("/")
    def get_root():
        return RedirectResponse(url="/ui")

    @app.get("/ui/static/fonts/system-ui/system-ui-Regular.woff2")
    @app.get("/ui/static/fonts/ui-sans-serif/ui-sans-serif-Regular.woff2")
    @app.get("/favicon.ico")
    def get_font():
        # remove confusing error
        return {}

    css = """
    .app {
        @media screen and (max-width: 768px) {
            padding: 8px !important;
        }
    }
    footer {
        display: none !important;
    }
    """
    with gr.Blocks(css=css) as gradio_block:
        with gr.Column():
            with gr.Group() as rtc_container:
                pass
    app = gr.mount_gradio_app(app, gradio_block, "/ui")

    chat_engine = ChatEngine()
    chat_engine.initialize(engine_config, app=app, ui=gradio_block, parent_block=rtc_container)

    return app
