import modal
import os

achatbot_version = os.getenv("ACHATBOT_VERSION", "0.0.10")
kimi_voice_img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "clang", "cmake", "ninja-build")
    .pip_install(
        [
            "achatbot["
            "fastapi_bot_server,"
            "livekit,livekit-api,daily,agora,"
            "silero_vad_analyzer,asr_processor,"
            "llm_transformers_manual_voice_kimi,"
            "queue"
            f"]=={achatbot_version}",
        ],
        extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
    )
    .pip_install("wheel")
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    .run_commands(
        "git clone -b feat/achatbot https://github.com/weedge/Kimi-Audio.git",
        "cd /Kimi-Audio && git pull origin feat/achatbot",
        "cd /Kimi-Audio && git checkout e6ef56fd7466feae8d4ea4341631e23dc5e63853",
    )
    .env(
        {
            "ACHATBOT_PKG": "1",
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "info"),
            "LLM_MODEL_NAME_OR_PATH": f"/root/.achatbot/models/{os.getenv('LLM_MODEL_NAME_OR_PATH', 'moonshotai/Kimi-Audio-7B-Instruct')}",
            # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TQDM_DISABLE": "1",
            "ALIAS_FREE_ACTIVATION_CUDA_SRC_PATH": "/Kimi-Audio/kimia_infer/models/detokenizer/vocoder/alias_free_activation/cuda",
            "MEL_FILTERS_PATH": "/Kimi-Audio/kimia_infer/models/tokenizer/whisper_Lv3/mel_filters.npz",
        }
    )
    # .pip_install(
    #    [
    #        "achatbot==0.0.10",
    #    ],
    #    extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
    # )
)


# ----------------------- app -------------------------------
app = modal.App("fastapi_webrtc_kimi_voice_bot")

HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)
TORCH_CACHE_DIR = "/root/.cache/torch"
torch_cache_vol = modal.Volume.from_name("torch_cache", create_if_missing=True)


# 128 MiB of memory and 0.125 CPU cores by default container runtime
@app.cls(
    image=kimi_voice_img,
    gpu=os.getenv("IMAGE_GPU", None),
    secrets=[modal.Secret.from_name("achatbot")],
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_dir,
        torch_cache_vol: torch_cache_vol,
    },
    cpu=2.0,
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
)
@modal.concurrent(max_inputs=int(os.getenv("IMAGE_CONCURRENT_CN", "1")))  # inputs per container
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
