import os
import modal

app = modal.App("tts-grpc")

achatbot_version = os.getenv("ACHATBOT_VERSION", "0.0.9.post3")

tts_grpc_image = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git", "git-lfs", "openmpi-bin", "libopenmpi-dev", "wget"
    )  # OpenMPI for distributed communication
    .pip_install(
        f"achatbot[{os.getenv('TTS_TAG', 'tts_generator_spark')},grpc]=={achatbot_version}",
        extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
    )
    .pip_install(
        f"achatbot[trtllm]=={achatbot_version}",
        pre=True,
        extra_index_url="https://pypi.nvidia.com",
    )
    .env(
        {
            "ACHATBOT_PKG": "1",
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "info"),
            "TORCH_CUDA_ARCH_LIST": "8.0 8.6 8.7 8.9 9.0",
            "PORT": os.getenv("GRPC_PORT", "50052"),
            "TLLM_LLMAPI_BUILD_CACHE": "1",
            "TTS_TAG": os.getenv("TTS_TAG", "tts_generator_spark"),
        }
    )
    # .pip_install("numpy==1.26.4", "transformers==4.48.3")
)

HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)
TRT_MODEL_DIR = "/root/.achatbot/trt_models"
trt_model_vol = modal.Volume.from_name("triton_trtllm_models", create_if_missing=True)

TRT_MODEL_CACHE_DIR = "/tmp/.cache/tensorrt_llm/llmapi/"
trt_model_cache_vol = modal.Volume.from_name("triton_trtllm_cache_models", create_if_missing=True)


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    cpu=2.0,
    image=tts_grpc_image,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_dir,
        TRT_MODEL_CACHE_DIR: trt_model_cache_vol,  # for auto trtllm-build engine
        TRT_MODEL_DIR: trt_model_vol,  # for mannual trtllm-build engine, use runner
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
TTS_TAG=tts_generator_spark IMAGE_GPU=L4 modal run src/tts/grpc_trtllm_tts_serve.py

"""


@app.local_entrypoint()
def main():
    run.remote()
