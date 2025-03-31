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
        f"achatbot[sglang]=={achatbot_version}",
        extra_options="--find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer/",
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.5/",
    )
    .env(
        {
            "ACHATBOT_PKG": "1",
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "debug"),
            "TORCH_CUDA_ARCH_LIST": "7.5 8.0 8.6 8.7 8.9 9.0",
        }
    )
    # .pip_install("numpy==1.26.4", "transformers==4.48.3")
)

HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    cpu=2.0,
    image=tts_grpc_image,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_dir,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=100,
)
def run():
    import subprocess

    with modal.forward(50052, unencrypted=True) as tunnel:
        print(
            f"use tunnel.tcp_socket = {tunnel.tcp_socket[0]}:{tunnel.tcp_socket[1]} to connect tritonserver with tcp(grpc)"
        )
        cmd = f"python -m achatbot.cmd.grpc.speaker.server.serve"
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)


"""
# run tts grpc serve with llm generator by tcp tunnel

# tts_generator_spark
TTS_TAG=tts_generator_spark IMAGE_GPU=T4 modal run src/tts/grpc_sglang_tts_serve.py

"""


@app.local_entrypoint()
def main():
    run.remote()
