import modal
import os


ACHATBOT_VERSION = os.getenv("ACHATBOT_VERSION", "0.0.24.post5")
IMAGE_GPU = os.getenv("IMAGE_GPU", "L4")
img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "docker.cnb.cool/tencent/hunyuan/hunyuan-7b:hunyuan-7b-trtllm",
        add_python="3.12",  # modal install /usr/local/bin/python3.12.1 or 3.10.13
    )
    .entrypoint([])  # remove verbose logging by base image on entry
    .run_commands(
        "/usr/local/bin/python --version",
        "/usr/bin/python --version",
        "echo $PATH",
        "/usr/bin/pip list",
        "update-alternatives --install /usr/local/bin/python python3 /usr/local/bin/python3.12 1",
        "update-alternatives --install /usr/local/bin/python python3 /usr/bin/python3.12 2",
        "python --version",
        "pip list",
    )
    # https://nvidia.github.io/TensorRT-LLM/latest/installation/build-from-source-linux.html
    # .run_commands(
    #    "git lfs install",
    #    "git clone https://github.com/NVIDIA/TensorRT-LLM.git",
    #    "cd TensorRT-LLM && git checkout 064eb7a70f29f45a74b5b080aafd0f6a872ed4b5",
    #    "cd TensorRT-LLM && pip install -r requirements.txt",
    # )
    .env(
        {
            # "TQDM_DISABLE": "1",
            "LLM_MODEL": os.getenv("LLM_MODEL", "tencent/Hunyuan-MT-7B"),
            "LD_LIBRARY_PATH": "/usr/local/tensorrt/targets/x86_64-linux-gnu/lib:$LD_LIBRARY_PATH",
        }
    )
    .run_commands(
        "cat /modal_requirements.txt",
        "pip list",
        "pip install -r /modal_requirements.txt",
    )
    .pip_install(
        "achatbot["
        "fastapi_bot_server,"
        "silero_vad_analyzer,"
        "sense_voice_asr,deepgram_asr_processor,"
        "tts_edge,"
        "queue"
        f"]=={ACHATBOT_VERSION}",
        extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
    )
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        "torchaudio==2.7.1",
    )
    .pip_install(
        "fastapi==0.115.4",
        "pydantic==2.9.1",
        "cloudpickle>=3.0.0",
        "protobuf==4.24.4",
    )
    .pip_install("onnxruntime", "funasr_onnx")
    .apt_install("ffmpeg")
    # .apt_install("openmpi-bin", "libopenmpi-dev") # no mpi
    .env(
        {
            "ACHATBOT_PKG": "1",
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "info"),
            "CONFIG_FILE": os.getenv(
                "CONFIG_FILE",
                "/root/.achatbot/config/bots/fastapi_websocket_asr_translate_trtllm-pytorch-hunyuan-mt_tts_bot.json",
            ),
            # "TQDM_DISABLE": "1",
            # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
        }
    )
)

# img = img.pip_install(
#    f"achatbot==0.0.24.post56",
#    extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://test.pypi.org/simple/"),
# )


HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)
TORCH_CACHE_DIR = "/root/.cache/torch"
torch_cache_vol = modal.Volume.from_name("torch_cache", create_if_missing=True)
CONFIG_DIR = "/root/.achatbot/config"
config_vol = modal.Volume.from_name("config", create_if_missing=True)
VLLM_CACHE_DIR = "/root/.cache/vllm"
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


# ----------------------- app -------------------------------
app = modal.App("fastapi_ws_translate-hunyuan-mt_bot")


# 128 MiB of memory and 0.125 CPU cores by default container runtime
@app.cls(
    image=img,
    gpu=IMAGE_GPU,
    secrets=[modal.Secret.from_name("achatbot")],
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_dir,
        TORCH_CACHE_DIR: torch_cache_vol,
        CONFIG_DIR: config_vol,
        VLLM_CACHE_DIR: vllm_cache_vol,
    },
    cpu=2.0,
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
    # allow_concurrent_inputs=int(os.getenv("IMAGE_CONCURRENT_CN", "1")),
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
        from achatbot.cmd.websocket.server.fastapi_ws_bot_serve import app as fastapi_app

        return fastapi_app


"""
modal volume create config

modal volume put config ./config/bots/fastapi_websocket_asr_translate_trtllm-pytorch-hunyuan-mt_tts_bot.json /bots/ -f

IMAGE_GPU=L4 ACHATBOT_VERSION=0.0.24.post5 \
    CONFIG_FILE=/root/.achatbot/config/bots/fastapi_websocket_asr_translate_trtllm-pytorch-hunyuan-mt_tts_bot.json \
    modal serve src/fastapi_ws_translate-hunyuan-trtllm_bot_serve.py


# cold start fastapi websocket server
curl -v -XGET "https://weedge--fastapi-ws-translate-hunyuan-mt-bot-srv-app-dev.modal.run/health"

# run websocket ui
cd ui/websocket && python -m http.server
# - access http://localhost:8000/translation   
# - change url to wss://weedge--fastapi-ws-translate-hunyuan-mt-bot-srv-app-dev.modal.run
# - click `Start Audio` to speech translation with Translation bot

"""
