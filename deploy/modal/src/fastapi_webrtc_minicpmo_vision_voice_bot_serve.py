import modal
import os


class ContainerRuntimeConfig:
    images = {
        "default": (
            # https://hub.docker.com/r/pytorch/pytorch/tags
            modal.Image.from_registry(
                "pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel", add_python="3.11"
            )
            .apt_install("git", "git-lfs", "ffmpeg")
            .env(
                {
                    "HF_HUB_ENABLE_HF_TRANSFER": "1",
                    "ACHATBOT_PKG": "1",
                    "LOG_LEVEL": os.getenv("LOG_LEVEL", "info"),
                    "IMAGE_NAME": os.getenv("IMAGE_NAME", "default"),
                    "USE_GPTQ_CKPT": os.getenv("USE_GPTQ_CKPT", ""),
                    "LLM_MODEL_NAME_OR_PATH": f"/root/.achatbot/models/{os.getenv('LLM_MODEL_NAME_OR_PATH', 'openbmb/MiniCPM-o-2_6')}",
                    # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
                    # nvcc --list-gpu-arch
                    "TORCH_CUDA_ARCH_LIST": "7.5;8.0;8.6+PTX;8.9;9.0",  # for auto-gptq install
                }
            )
            .run_commands(
                [
                    "pip install --no-build-isolation achatbot[flash-attn]",
                    "if [ $USE_GPTQ_CKPT ];then"
                    + " git clone https://github.com/OpenBMB/AutoGPTQ.git -b minicpmo"
                    + " && cd AutoGPTQ"
                    + " && pip install -vvv --no-build-isolation -e . "
                    + ";fi",
                ]
            )
            .pip_install(
                [
                    "achatbot["
                    "fastapi_bot_server,"
                    "livekit,livekit-api,daily,agora,"
                    "silero_vad_analyzer,"
                    "llm_transformers_manual_vision_voice_minicpmo,"
                    "queue"
                    "]~=0.0.9.post10",
                    "huggingface_hub[hf_transfer]==0.26.0",
                    "wget",
                ],
                extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
                # extra_index_url="https://test.pypi.org/simple/",
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
        app_name = "fastapi_webrtc_omni_bot"
        if image_name != "default":
            app_name = f"fastapi_webrtc_omni_{image_name}_bot"
        print(f"app_name:{app_name}")
        return app_name

    @staticmethod
    def get_gpu():
        # T4, L4, A10G, A100, H100
        gpu = os.getenv("IMAGE_GPU", None)
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

# ----------------------- app -------------------------------
app = modal.App("fastapi_webrtc_minicpmo_omni_bot")

# volume = modal.Volume.from_name("bot_config", create_if_missing=True)


# 128 MiB of memory and 0.125 CPU cores by default container runtime
@app.cls(
    image=ContainerRuntimeConfig.get_img(),
    # volumes={"/bots": volume},
    gpu=ContainerRuntimeConfig.get_gpu(),
    secrets=[modal.Secret.from_name("achatbot")],
    cpu=2.0,
    scaledown_window=300,
    timeout=600,
    # allow_concurrent_inputs=ContainerRuntimeConfig.get_allow_concurrent_inputs(),
)
@modal.concurrent(max_inputs=int(os.getenv("IMAGE_CONCURRENT_CN", "1")))  # inputs per container
class Srv:
    @modal.build()
    def setup(self):
        import wget

        # https://huggingface.co/docs/huggingface_hub/guides/download
        from huggingface_hub import snapshot_download
        from achatbot.common.types import MODELS_DIR, ASSETS_DIR

        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)

        # ref audio
        os.makedirs(ASSETS_DIR, exist_ok=True)
        print(f"start downloading assets to dir:{ASSETS_DIR}")
        assets = [
            "https://raw.githubusercontent.com/ai-bot-pro/achatbot/refs/heads/main/test/audio_files/asr_example_zh.wav"
        ]
        for url in assets:
            wget.download(url, out=ASSETS_DIR)

        os.makedirs(MODELS_DIR, exist_ok=True)
        logging.info(f"start downloading model to dir:{MODELS_DIR}")

        # asr model repo
        if "sense_voice_asr" in os.getenv("ASR_TAG", "sense_voice_asr"):
            local_dir = os.path.join(MODELS_DIR, "FunAudioLLM/SenseVoiceSmall")
            snapshot_download(
                repo_id="FunAudioLLM/SenseVoiceSmall",
                repo_type="model",
                allow_patterns="*",
                local_dir=local_dir,
            )
            logging.info(f"sense_voice_asr model to dir:{local_dir} done")

        # llm model repo
        llm_model_repo = os.getenv("LLM_MODEL_NAME_OR_PATH")
        if llm_model_repo:
            repo_id = "/".join(llm_model_repo.split("/")[-2:])
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                allow_patterns="*",
                local_dir=os.path.join(MODELS_DIR, repo_id),
            )

        if os.getenv("USE_GPTQ_CKPT"):
            repo_id = "openbmb/MiniCPM-o-2_6-int4"
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                allow_patterns="*",
                local_dir=os.path.join(MODELS_DIR, repo_id),
            )
            logging.info("download gptq int4 model done")

        logging.info("download model done")

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
