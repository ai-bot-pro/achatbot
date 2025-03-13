import os
import modal

app = modal.App("tritonserver")

tritonserver_image = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
    modal.Image.from_registry(
        "nvcr.io/nvidia/tritonserver:25.02-trtllm-python-py3",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "cmake", "nvidia-cuda-dev")
    .pip_install(
        "torch",
        "einx==0.3.0",
        "omegaconf==2.3.0",
        "soundfile==0.12.1",
        "soxr==0.5.0.post1",
        "transformers==4.46.2",
        "librosa",
        # extra_index_url="https://pypi.nvidia.com",
    )
    .run_commands(
        # "git clone https://github.com/pytorch/audio.git",
        # "cd audio && git checkout c670ad8 && PATH=/usr/local/cuda/bin:$PATH python3 setup.py develop",
        "git clone https://github.com/SparkAudio/Spark-TTS.git",
    )
    .env(
        {
            "PYTHONPATH": os.getenv("IMAGE_PYTHONPATH", "/Spark-TTS/"),
        }
    )
)


HF_MODEL_DIR = "/root/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
TRT_MODEL_DIR = "/root/trt_models"
trt_model_vol = modal.Volume.from_name("triton_trtllm_models", create_if_missing=True)
TRITONSERVER_DIR = "/root/tritonserver"
tritonserver_vol = modal.Volume.from_name("tritonserver", create_if_missing=True)


# see: https://github.com/triton-inference-server/python_backend/blob/main/README.md
@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    retries=0,
    image=tritonserver_image,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        TRT_MODEL_DIR: trt_model_vol,
        TRITONSERVER_DIR: tritonserver_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
)
def serve(app_name: str) -> str:
    import subprocess

    cmd = f"which nvcc".split(" ")
    subprocess.run(cmd, cwd="/", check=True)
    cmd = f"nvcc --version".split(" ")
    subprocess.run(cmd, cwd="/", check=True)

    cmd = f"which tritonserver".split(" ")
    subprocess.run(cmd, cwd="/")

    # cmd = f"ls -lh /opt/tritonserver/bin/".split(" ")
    # subprocess.run(cmd, cwd="/")

    # cmd = f"ls -lh /opt/tritonserver/backends/".split(" ")
    # subprocess.run(cmd, cwd="/")

    # cmd = f"ls -lh /opt/tritonserver/backends/tensorrtllm/".split(" ")
    # subprocess.run(cmd, cwd="/")

    cmd = f"ldd /opt/tritonserver/backends/tensorrtllm/libtriton_tensorrtllm.so".split(" ")
    subprocess.run(cmd, cwd="/")

    model_repo = os.path.join(TRITONSERVER_DIR, app_name)
    cmd = f"ls -lh {model_repo}".split(" ")
    subprocess.run(cmd, cwd="/")

    cmd = f"tritonserver --model-repository {model_repo}"
    print(cmd)
    subprocess.run(cmd.split(" "), cwd="/")


"""
# run tritonserver
modal run src/llm/trtllm/bench/tritonserver.py --app-name tts-spark
"""


@app.local_entrypoint()
def main(app_name: str = ""):
    serve.remote(app_name)
