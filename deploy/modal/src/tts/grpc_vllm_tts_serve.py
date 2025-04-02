import os
import modal

app = modal.App("tts-grpc")

achatbot_version = os.getenv("ACHATBOT_VERSION", "0.0.9.post3")

tts_grpc_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "git-lfs", "ffmpeg")
    .pip_install(
        f"achatbot[{os.getenv('TTS_TAG', 'tts_generator_spark')},grpc]=={achatbot_version}",
        extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
    )
    .pip_install(
        f"achatbot[vllm]=={achatbot_version}",
        "flashinfer-python==0.2.0.post2",
        extra_options="--find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer/",
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.5/",
    )
    .env(
        {
            "ACHATBOT_PKG": "1",
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "info"),
            "TORCH_CUDA_ARCH_LIST": "8.0 8.6 8.7 8.9 9.0",
            "VLLM_USE_V1": os.getenv("VLLM_USE_V1", "1"),
            "PORT": os.getenv("GRPC_PORT", "50052"),
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "TTS_TAG": os.getenv("TTS_TAG", "tts_generator_spark"),
        }
    )
    # .pip_install("numpy==1.26.4", "transformers==4.48.3")
)

HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)
VLLM_CACHE_DIR = "/root/.cache/vllm"
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    cpu=2.0,
    image=tts_grpc_image,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_dir,
        VLLM_CACHE_DIR: vllm_cache_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=100,
)
def run():
    from achatbot.cmd.grpc.speaker.server.serve import serve

    port = int(os.getenv("PORT", "50052"))
    with modal.forward(port, unencrypted=True) as tunnel:
        print(
            f"use tunnel.tcp_socket = {tunnel.tcp_socket[0]}:{tunnel.tcp_socket[1]} to connect tritonserver with tcp(grpc)"
        )
        serve()


"""
# run tts grpc serve with llm generator by tcp tunnel

# tts_generator_spark
TTS_TAG=tts_generator_spark IMAGE_GPU=L4 modal run src/tts/grpc_vllm_tts_serve.py

"""


@app.local_entrypoint()
def main():
    run.remote()
