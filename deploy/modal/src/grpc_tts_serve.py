import os
import modal

app = modal.App("tts-grpc")
achatbot_version = os.getenv("ACHATBOT_VERSION", "0.0.23")

tts_grpc_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "ffmpeg")
    .pip_install(
        f"achatbot[{os.getenv('TTS_TAG', 'tts_edge')},grpc]=={achatbot_version}",
        extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
    )
    .pip_install(
        "protobuf==5.29.2",
    )
    .env(
        {
            "ACHATBOT_PKG": "1",
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "info"),
        }
    )
    # .pip_install("numpy==1.26.4", "transformers==4.48.3")
)

HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)
TORCH_CACHE_DIR = "/root/.cache/torch"
torch_cache_vol = modal.Volume.from_name("torch_cache", create_if_missing=True)


@app.function(
    gpu=os.getenv("IMAGE_GPU", None),
    cpu=2.0,
    image=tts_grpc_image,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_dir,
        TORCH_CACHE_DIR: torch_cache_vol,
    },
    timeout=86400,  # default 300s
    scaledown_window=1200,
    max_containers=10,
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
# run tts grpc serve  by tcp tunnel
# default tts_edge
modal run src/grpc_tts_serve.py 

# tts_f5
TTS_TAG=tts_f5 IMAGE_GPU=T4 modal run src/grpc_tts_serve.py

# tts_spark
TTS_TAG=tts_spark IMAGE_GPU=T4 modal run src/grpc_tts_serve.py

# tts_orpheus 
TTS_TAG=tts_orpheus IMAGE_GPU=T4 modal run src/grpc_tts_serve.py

# tts_mega3
TTS_TAG=tts_mega3 IMAGE_GPU=T4 modal run src/grpc_tts_serve.py

# tts_higgs
TTS_TAG=tts_higgs IMAGE_GPU=L4 modal run src/grpc_tts_serve.py

# run grpc client
TTS_TAG=tts_higgs IS_SAVE=1 SERVE_ADDR=r447.modal.host:35591 \
    TTS_AUDIO_TOKENIZER_PATH=/root/.achatbot/models/bosonai/higgs-audio-v2-tokenizer \
    TTS_LM_MODEL_PATH=/root/.achatbot/models/bosonai/higgs-audio-v2-generation-3B-base \
    TTS_REF_TEXT="对，这就是我，万人敬仰的太乙真人。" \
    TTS_REF_AUDIO_PATH="/root/.achatbot/assets/basic_ref_zh.wav" \
    TTS_CHUNK_SIZE=16 \
    python -m src.cmd.grpc.speaker.client

# tips: u can pip install achatbot to run grpc client
ACHATBOT_PKG=1 TTS_TAG=tts_higgs IS_SAVE=1 SERVE_ADDR=r447.modal.host:35591 \
    TTS_AUDIO_TOKENIZER_PATH=/root/.achatbot/models/bosonai/higgs-audio-v2-tokenizer \
    TTS_LM_MODEL_PATH=/root/.achatbot/models/bosonai/higgs-audio-v2-generation-3B-base \
    TTS_REF_TEXT="对，这就是我，万人敬仰的太乙真人。" \
    TTS_REF_AUDIO_PATH="/root/.achatbot/assets/basic_ref_zh.wav" \
    TTS_CHUNK_SIZE=16 \
    python -m achatbot.cmd.grpc.speaker.client
"""


@app.local_entrypoint()
def main():
    run.remote()
