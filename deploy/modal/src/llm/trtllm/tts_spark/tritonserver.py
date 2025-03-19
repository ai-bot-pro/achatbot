import os
import modal

app = modal.App("tritonserver")

tritonserver_image = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
    # default install python3.12.3, in /usr/bin/python3.12.3
    modal.Image.from_registry(
        "nvcr.io/nvidia/tritonserver:25.02-trtllm-python-py3",
        add_python="3.12",  # modal install /usr/local/bin/python3.12.1
    )
    .apt_install(
        "tree",
        "git",
        "git-lfs",
        "cmake",
        "rapidjson-dev",
        "libarchive-dev",
        "zlib1g-dev",
    )
    .run_commands(
        # "update-alternatives --install /usr/local/bin/python python3 /usr/local/bin/python3.12 1",
        # "update-alternatives --install /usr/local/bin/python python3 /usr/bin/python3.12 2",
        # "python --version",
    )
    .run_commands(
        "cmake --version",  # cmake>=3.17
        "python --version",
        "mkdir -p /stub",
        "git clone https://github.com/triton-inference-server/python_backend -b r25.02",
        "cd python_backend && mkdir build",
        "cd /python_backend/build && cmake -DPYBIND11_FINDPYTHON=ON -DPYTHON_EXECUTABLE=$(which python) -DTRITON_ENABLE_GPU=ON -DTRITON_BACKEND_REPO_TAG=r25.02 -DTRITON_COMMON_REPO_TAG=r25.02 -DTRITON_CORE_REPO_TAG=r25.02 -DCMAKE_INSTALL_PREFIX:PATH=/stub/ .. && make triton-python-backend-stub",
    )
    .run_commands(
        f"git clone https://github.com/weedge/Spark-TTS.git -b {os.getenv('TAG_OR_HASH', 'feat/runtime-stream')}",
        "pip install -r Spark-TTS/requirements.txt",
        # "pip install torch==2.6.0 torchaudio==2.6.0 transformers safetensors==0.5.2 einx==0.3.0 omegaconf==2.3.0 soundfile==0.12.1 soxr==0.5.0.post1",
    )
    .env(
        {
            "PYTHONPATH": os.getenv("IMAGE_PYTHONPATH", "/Spark-TTS/"),
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES", "0"),
            "PYTHONIOENCODING": "utf-8",
            "APP_NAME": os.getenv("APP_NAME", "tts-spark"),
            "STREAM": os.getenv("STREAM", ""),
        }
    )
)


HF_MODEL_DIR = "/root/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
TRT_MODEL_DIR = "/root/trt_models"
trt_model_vol = modal.Volume.from_name("triton_trtllm_models", create_if_missing=True)
TRITONSERVER_DIR = "/root/tritonserver"
tritonserver_vol = modal.Volume.from_name("tritonserver", create_if_missing=True)


# ⭐️ https://github.com/triton-inference-server/python_backend/blob/main/README.md
@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    image=tritonserver_image,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        TRT_MODEL_DIR: trt_model_vol,
        TRITONSERVER_DIR: tritonserver_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=100,
)
def run():
    import subprocess

    print("--" * 20)
    cmd = f"ldd /opt/tritonserver/backends/python/triton_python_backend_stub"
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)

    print("--" * 20)
    cmd = f"ldd /python_backend/build/triton_python_backend_stub"
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)

    app_name: str = os.getenv("APP_NAME", "tts-spark")
    model_repo = os.path.join(TRITONSERVER_DIR, app_name)

    cmd = f"cp /python_backend/build/triton_python_backend_stub {model_repo}/audio_tokenizer"
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)

    spark_dir = "spark_tts"
    if bool(os.getenv("STREAM", "")):
        spark_dir = "spark_tts_decoupled"
    cmd = f"cp /python_backend/build/triton_python_backend_stub {model_repo}/{spark_dir}"
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)

    cmd = f"cp /python_backend/build/triton_python_backend_stub {model_repo}/vocoder"
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)
    cmd = f"tree {model_repo}"
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)

    with modal.forward(8001, unencrypted=True) as tunnel:
        print(
            f"use tunnel.tcp_socket = {tunnel.tcp_socket[0]}:{tunnel.tcp_socket[1]} to connect tritonserver with tcp(grpc)"
        )
        cmd = f"tritonserver --model-repository {model_repo} --log-verbose 0 --log-info True"
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)


# ⭐️ https://github.com/triton-inference-server/python_backend/blob/main/README.md
@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    image=tritonserver_image,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        TRT_MODEL_DIR: trt_model_vol,
        TRITONSERVER_DIR: tritonserver_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=100,
)
@modal.web_server(port=8000, startup_timeout=10 * 60)
def serve():
    import subprocess

    print("--" * 20)
    cmd = f"ldd /opt/tritonserver/backends/python/triton_python_backend_stub"
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)

    print("--" * 20)
    cmd = f"ldd /python_backend/build/triton_python_backend_stub"
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)

    app_name: str = os.getenv("APP_NAME", "tts-spark")
    model_repo = os.path.join(TRITONSERVER_DIR, app_name)

    cmd = f"cp /python_backend/build/triton_python_backend_stub {model_repo}/audio_tokenizer"
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)
    spark_dir = "spark_tts"
    if bool(os.getenv("STREAM", "")):
        spark_dir = "spark_tts_decoupled"
    cmd = f"cp /python_backend/build/triton_python_backend_stub {model_repo}/{spark_dir}"
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)
    cmd = f"cp /python_backend/build/triton_python_backend_stub {model_repo}/vocoder"
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)
    cmd = f"tree {model_repo}"
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)

    cmd = f"tritonserver --model-repository {model_repo}"
    print(cmd)
    subprocess.Popen(cmd, shell=True)


"""
# run tritonserver
APP_NAME=tts-spark modal serve src/llm/trtllm/tts_spark/tritonserver.py 
STREAM=1 APP_NAME=tts-spark modal serve src/llm/trtllm/tts_spark/tritonserver.py 

# curl server is ready
curl -vv -X GET "https://weedge--tritonserver-serve-dev.modal.run/v2/health/ready" -H  "accept: application/json"

# run grpc tritonserver by tcp tunnel and http server
APP_NAME=tts-spark modal run src/llm/trtllm/tts_spark/tritonserver.py 
STREAM=1 APP_NAME=tts-spark modal run src/llm/trtllm/tts_spark/tritonserver.py 
"""


@app.local_entrypoint()
def main():
    run.remote()
