import modal
import os

achatbot_version = os.getenv("ACHATBOT_VERSION", "0.0.24")
LLM_TAG = os.getenv("LLM_TAG", "llm_ctranslate2_generator")
vllm_version = os.getenv("VLLM_VERSION", "0.8.0")
transformers_version = os.getenv("TRANSFORMERS_VERSION", "4.51.3")
img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "clang", "cmake")
    .pip_install("wheel")
    .pip_install(
        [
            "achatbot["
            "fastapi_bot_server,"
            "silero_vad_analyzer,"
            "sense_voice_asr,deepgram_asr_processor,"
            "tts_edge,"
            "queue"
            f"]=={achatbot_version}",
        ],
        extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
    )
    .pip_install("onnxruntime", "funasr_onnx")
)

if LLM_TAG == "llm_ctranslate2_generator":
    img = img.pip_install(
        "ctranslate2",
        "transformers[torch]",
    )
if LLM_TAG == "llm_vllm_generator":
    img = img.pip_install(
        f"vllm=={vllm_version}",
        f"transformers=={transformers_version}",
    ).env(
        {
            # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
            "TORCH_CUDA_ARCH_LIST": "8.0 8.6 8.9 9.0 9.0a 10.0",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "VLLM_USE_V1": os.getenv("VLLM_USE_V1", "1"),
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )
if LLM_TAG == "llm_sglang_generator":
    img = (
        img.pip_install(  # add sglang and some Python dependencies
            # as per sglang website: https://docs.sglang.ai/get_started/install.html
            "flashinfer-python",
            "sglang[all]>=0.5.1.post1",
            extra_options="--find-links https://flashinfer.ai/whl/cu126/torch2.6/flashinfer-python/",
            extra_index_url="https://flashinfer.ai/whl/cu126/torch2.6/",
        )
        .apt_install("libnuma-dev")
        .pip_install(f"transformers==4.56.0")
        .env(
            {
                # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
                "TORCH_CUDA_ARCH_LIST": "7.5 8.0 8.6 8.9 9.0 9.0a 10.0",
            }
        )
    )

if LLM_TAG in ["llm_trtllm_generator", "llm_trtllm_runner_generator"]:
    GIT_TAG_OR_HASH = os.getenv("GIT_TAG_OR_HASH", "0.18.0")  # 0.18.0 don't support 10.0a+
    img = (
        img.entrypoint([])  # remove verbose logging by base image on entry
        .apt_install("openmpi-bin", "libopenmpi-dev")
        .pip_install(
            f"tensorrt-llm=={GIT_TAG_OR_HASH}",
            "pynvml<12",  # avoid breaking change to pynvml version API
            "flashinfer-python==0.2.5",
            "cuda-python==12.9.1",
            pre=True,
            extra_index_url="https://pypi.nvidia.com",
        )
        # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
        .env({"TORCH_CUDA_ARCH_LIST": "8.0 8.9 9.0 9.0a 10.0"})
    )

if LLM_TAG == "llm_transformers_generator":
    img = (
        img.pip_install("accelerate", "torch==2.6.0", "transformers==4.56.0")
        # https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1
        .pip_install("flash-attn==2.7.4.post1", extra_options="--no-build-isolation")
        .pip_install("compressed-tensors==0.11.0")
    )

# img = img.pip_install(
#   f"achatbot==0.0.24.post51",
#   extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://test.pypi.org/simple/"),
# )

img = img.env(
    {
        "ACHATBOT_PKG": "1",
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "info"),
        "CONFIG_FILE": os.getenv(
            "CONFIG_FILE",
            "/root/.achatbot/config/bots/fastapi_websocket_asr_translate_ctranslate2_tts_bot.json",
            # "/root/.achatbot/config/bots/fastapi_websocket_asr_translate_transformers_tts_bot.json"
            # "/root/.achatbot/config/bots/fastapi_websocket_asr_translate_vllm_tts_bot.json",
            # "/root/.achatbot/config/bots/fastapi_websocket_asr_translate_sglang_tts_bot.json",
            # "/root/.achatbot/config/bots/fastapi_websocket_asr_translate_trtllm_tts_bot.json",
            # "/root/.achatbot/config/bots/fastapi_websocket_asr_translate_trtllm_runner_tts_bot.json",
        ),
        # "TQDM_DISABLE": "1",
        # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
    }
)


# ----------------------- app -------------------------------
app = modal.App("fastapi_ws_translate_bot")

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


# 128 MiB of memory and 0.125 CPU cores by default container runtime
@app.cls(
    image=img,
    gpu=os.getenv("IMAGE_GPU", None),
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

# - Seed-X

modal volume put config ./config/bots/fastapi_websocket_asr_translate_transformers_tts_bot.json /bots/ -f

IMAGE_GPU=L4 LLM_TAG=llm_transformers_generator \
    ACHATBOT_VERSION=0.0.24.post2 \
    CONFIG_FILE=/root/.achatbot/config/bots/fastapi_websocket_asr_translate_transformers_tts_bot.json \
    modal serve src/fastapi_ws_translate_bot_serve.py

modal volume put config ./config/bots/fastapi_websocket_asr_translate_ctranslate2_tts_bot.json /bots/ -f

IMAGE_GPU=L4 LLM_TAG=llm_ctranslate2_generator \
    ACHATBOT_VERSION=0.0.24 \
    CONFIG_FILE=/root/.achatbot/config/bots/fastapi_websocket_asr_translate_ctranslate2_tts_bot.json \
    modal serve src/fastapi_ws_translate_bot_serve.py


modal volume put config ./config/bots/fastapi_websocket_asr_translate_vllm_tts_bot.json /bots/ -f

IMAGE_GPU=L4 LLM_TAG=llm_vllm_generator \
    ACHATBOT_VERSION=0.0.24 \
    CONFIG_FILE=/root/.achatbot/config/bots/fastapi_websocket_asr_translate_vllm_tts_bot.json \
    modal serve src/fastapi_ws_translate_bot_serve.py


modal volume put config ./config/bots/fastapi_websocket_asr_translate_sglang_tts_bot.json /bots/ -f

IMAGE_GPU=L4 LLM_TAG=llm_sglang_generator \
    ACHATBOT_VERSION=0.0.24 \
    CONFIG_FILE=/root/.achatbot/config/bots/fastapi_websocket_asr_translate_sglang_tts_bot.json \
    modal serve src/fastapi_ws_translate_bot_serve.py


modal volume put config ./config/bots/fastapi_websocket_asr_translate_trtllm_tts_bot.json /bots/ -f

IMAGE_GPU=L4 LLM_TAG=llm_trtllm_generator \
    ACHATBOT_VERSION=0.0.24.post1 \
    CONFIG_FILE=/root/.achatbot/config/bots/fastapi_websocket_asr_translate_trtllm_tts_bot.json \
    modal serve src/fastapi_ws_translate_bot_serve.py


modal volume put config ./config/bots/fastapi_websocket_asr_translate_trtllm_runner_tts_bot.json /bots/ -f

modal run src/llm/trtllm/translation/compile_model.py \
    --app-name "seed-x" \
    --hf-repo-dir "ByteDance-Seed/Seed-X-PPO-7B" \
    --trt-dtype "bfloat16" \
    --convert-other-args "" \
    --compile-other-args "--max_batch_size 16 --max_num_tokens 32768"

IMAGE_GPU=L4 LLM_TAG=llm_trtllm_runner_generator \
    ACHATBOT_VERSION=0.0.24.post1 \
    CONFIG_FILE=/root/.achatbot/config/bots/fastapi_websocket_asr_translate_trtllm_runner_tts_bot.json \
    modal serve src/fastapi_ws_translate_bot_serve.py
    


# - Hunyuan-MT

modal volume put config ./config/bots/fastapi_websocket_asr_translate-hunyuan-mt_transformers_tts_bot.json /bots/ -f

IMAGE_GPU=L4 LLM_TAG=llm_transformers_generator \
    ACHATBOT_VERSION=0.0.24.post2 \
    CONFIG_FILE=/root/.achatbot/config/bots/fastapi_websocket_asr_translate-hunyuan-mt_transformers_tts_bot.json \
    modal serve src/fastapi_ws_translate_bot_serve.py


modal volume put config ./config/bots/fastapi_websocket_asr_translate-hunyuan-mt_vllm_tts_bot.json /bots/ -f

IMAGE_GPU=L4 LLM_TAG=llm_vllm_generator \
    ACHATBOT_VERSION=0.0.24.post2 VLLM_VERSION=0.10.0 TRANSFORMERS_VERSION=4.56.0 \
    CONFIG_FILE=/root/.achatbot/config/bots/fastapi_websocket_asr_translate-hunyuan-mt_vllm_tts_bot.json \
    modal serve src/fastapi_ws_translate_bot_serve.py


modal volume put config ./config/bots/fastapi_websocket_asr_translate-hunyuan-mt_sglang_tts_bot.json /bots/ -f

IMAGE_GPU=L4 LLM_TAG=llm_sglang_generator \
    ACHATBOT_VERSION=0.0.24.post2 \
    CONFIG_FILE=/root/.achatbot/config/bots/fastapi_websocket_asr_translate-hunyuan-mt_sglang_tts_bot.json \
    modal serve src/fastapi_ws_translate_bot_serve.py


# cold start fastapi websocket server
curl -v -XGET "https://weedge--fastapi-ws-translate-bot-srv-app-dev.modal.run/health"

# run websocket ui
cd ui/websocket && python -m http.server
# - access http://localhost:8000/translation   
# - change url to wss://weedge--fastapi-ws-translate-bot-srv-app-dev.modal.run
# - click `Start Audio` to speech translation with Translation bot

"""
