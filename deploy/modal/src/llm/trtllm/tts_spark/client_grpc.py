import os
import modal

app = modal.App("trtllm")

trtllm_image = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
    modal.Image.from_registry(
        "nvcr.io/nvidia/tritonserver:25.02-trtllm-python-py3",
        # add_python="3.10",
    )
    .apt_install("git", "git-lfs", "cmake")
    .run_commands("pip list")
    .run_commands(
        # "pip install torch",
        "git clone https://github.com/pytorch/audio.git",
        "cd audio && git checkout c670ad8 && PATH=/usr/local/cuda/bin:$PATH python3 setup.py develop",
        "git clone https://github.com/SparkAudio/Spark-TTS.git",
    )
    .pip_install(
        "einx==0.3.0",
        "omegaconf==2.3.0",
        "soundfile==0.12.1",
        "soxr==0.5.0.post1",
        "transformers==4.46.2",
        "gradio",
        "tritonclient",
        "librosa",
        "huggingface_hub[hf_transfer]==0.26.2",
        # extra_index_url="https://pypi.nvidia.com",
    )
    .env({})  # faster model transfers
)

BENCH_DIR = "/data/bench"
bench_dir = modal.Volume.from_name("bench", create_if_missing=True)

models_vol = modal.Volume.from_name("models", create_if_missing=True)
triton_trtllm_vol = modal.Volume.from_name("triton_trtllm_models", create_if_missing=True)


@app.function(
    # gpu="L4",
    cpu=8.0,
    retries=0,
    image=trtllm_image,
    volumes={
        "/Spark-TTS/pretrained_models": models_vol,
        "/Spark-TTS/runtime/triton_trtllm": triton_trtllm_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
)
def compile_model() -> str:
    import subprocess

    cmd = "rm -rf Spark-TTS".split(" ")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/")
    print(result)

    cmd = "git clone https://github.com/SparkAudio/Spark-TTS.git".split(" ")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/")
    print(result)

    # triton_trtllm_vol.reload()

    cmd = "bash run.sh 0 2".split(" ")
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd="/Spark-TTS/runtime/triton_trtllm"
    )
    print(result)


@app.function(
    gpu="L4",
    retries=0,
    image=trtllm_image,
    timeout=1200,  # default 300s
    scaledown_window=1200,
)
def serve() -> str:
    import subprocess

    cmd = "ls -lh".split(" ")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/audio")
    print(result)


"""
# build image
modal run src/llm/trtllm/bench/tts_spark.py 
# run compile model from hf model to tensorrt model
modal run src/llm/trtllm/bench/tts_spark.py --mode compile_model
"""


@app.local_entrypoint()
def main(mode: str = ""):
    if mode == "compile_model":
        compile_model.remote()
    elif mode == "serve":
        serve.remote()
