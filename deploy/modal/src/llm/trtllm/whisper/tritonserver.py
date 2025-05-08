import os
import subprocess
import modal

app = modal.App("tritonserver")

# https://github.com/NVIDIA/TensorRT-LLM/blob/v0.20.0rc0/examples/models/core/whisper/README.md
# https://github.com/NVIDIA/TensorRT-LLM/tree/v0.18.0/examples/whisper
GIT_TAG_OR_HASH = os.getenv("GIT_TAG_OR_HASH", "v0.18.0")
TRITONSERVER_VERSION = "25.03"


tritonserver_image = (
    # https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/dockerfile/Dockerfile.trt_llm_backend
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
    # default install python3.12.3, in /usr/bin/python3.12.3
    modal.Image.from_registry(
        f"nvcr.io/nvidia/tritonserver:{TRITONSERVER_VERSION}-trtllm-python-py3",
        add_python="3.12",  # modal install /usr/local/bin/python3.12.1 or 3.10.13
    )
    .apt_install(
        "tree",
        "git",
        "git-lfs",
        "cmake",
        "rapidjson-dev",
        "libarchive-dev",
        "zlib1g-dev",
        "ffmpeg",
    )
    .run_commands(
        "which python",
        "cmake --version",  # cmake>=3.17
        "/usr/bin/python3 --version",
        "/usr/local/bin/python3 --version",
        "mkdir -p /stub",
        f"git clone https://github.com/triton-inference-server/python_backend -b r{TRITONSERVER_VERSION}",
        "cd python_backend && mkdir build",
        f"cd /python_backend/build && cmake -DPYBIND11_FINDPYTHON=ON -DPYTHON_EXECUTABLE=$(which python) -DTRITON_ENABLE_GPU=ON -DTRITON_BACKEND_REPO_TAG=r{TRITONSERVER_VERSION} -DTRITON_COMMON_REPO_TAG=r{TRITONSERVER_VERSION} -DTRITON_CORE_REPO_TAG=r{TRITONSERVER_VERSION} -DCMAKE_INSTALL_PREFIX:PATH=/stub/ .. && make triton-python-backend-stub",
    )
    .pip_install(
        # "numpy<2",
        "torch",
        "tiktoken",  # tokenizer
        "soundfile",  # wav file to np.array
        # "kaldialign", for client test WER
    )
    .pip_install(
        f"tensorrt-llm=={GIT_TAG_OR_HASH}",
        # "pynvml<12",  # avoid breaking change to pynvml version API for tensorrt_llm
        pre=True,
        extra_index_url="https://pypi.nvidia.com",
    )
    .env(
        {
            # "PYTHONPATH": os.getenv("IMAGE_PYTHONPATH", "/"),
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES", "0"),
            "PYTHONIOENCODING": "utf-8",
            "APP_NAME": os.getenv("APP_NAME", "whisper"),
            "TENSORRT_LLM_MODEL_NAME": os.getenv(
                "TENSORRT_LLM_MODEL_NAME", "whisper_bls,whisper_tensorrt_llm"
            ),
            "STREAM": os.getenv("STREAM", ""),
        }
    )
)


ASSETS_DIR = "/root/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)
TRT_MODEL_DIR = "/root/trt_models"
trt_model_vol = modal.Volume.from_name("triton_trtllm_models", create_if_missing=True)
TRITONSERVER_DIR = "/root/tritonserver"
tritonserver_vol = modal.Volume.from_name("tritonserver", create_if_missing=True)


# ⭐️ https://github.com/triton-inference-server/python_backend/blob/main/README.md
@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    image=tritonserver_image,
    volumes={
        ASSETS_DIR: assets_dir,
        TRT_MODEL_DIR: trt_model_vol,
        TRITONSERVER_DIR: tritonserver_vol,
    },
    timeout=3600,  # default 300s
    scaledown_window=1200,
    max_containers=int(os.getenv("MAX_CONTAINERS", "1")),
)
def run():
    app_name: str = os.getenv("APP_NAME", "whisper")
    model_repo = os.path.join(TRITONSERVER_DIR, app_name)
    prepare(model_repo)

    with modal.forward(8001, unencrypted=True) as tunnel:
        print(
            f"use tunnel.tcp_socket = {tunnel.tcp_socket[0]}:{tunnel.tcp_socket[1]} to connect tritonserver with tcp(grpc)",
        )
        with open(f"{model_repo}/tunnel_server_addr.txt", "w") as f:
            print(
                f"use tunnel.tcp_socket = {tunnel.tcp_socket[0]}:{tunnel.tcp_socket[1]} to connect tritonserver with tcp(grpc)",
                file=f,
            )
        cmd = f"tritonserver --model-repository {model_repo} --log-verbose 0 --log-info True "
        tensorrt_llm_model_name: str = os.getenv("TENSORRT_LLM_MODEL_NAME", "whisper")
        model_names = tensorrt_llm_model_name.split(",")
        if len(model_names) > 0:
            cmd += f"--model-control-mode=explicit "
        for model_name in model_names:
            cmd += f"--load-model {model_name} "
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)


# ⭐️ https://github.com/triton-inference-server/python_backend/blob/main/README.md
@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    image=tritonserver_image,
    volumes={
        ASSETS_DIR: assets_dir,
        TRT_MODEL_DIR: trt_model_vol,
        TRITONSERVER_DIR: tritonserver_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=100,
)
@modal.web_server(port=8000, startup_timeout=10 * 60)
def serve():
    app_name: str = os.getenv("APP_NAME", "whisper")
    model_repo = os.path.join(TRITONSERVER_DIR, app_name)
    prepare(model_repo)

    cmd = f"tritonserver --model-repository {model_repo} "
    tensorrt_llm_model_name: str = os.getenv("TENSORRT_LLM_MODEL_NAME", "whisper")
    model_names = tensorrt_llm_model_name.split(",")
    if len(model_names) > 0:
        cmd += f"--model-control-mode=explicit "
    for model_name in model_names:
        cmd += f"--load-model {model_name} "
    print(cmd)
    subprocess.Popen(cmd, shell=True)


def prepare(model_repo):
    import torch

    print("torch:", torch.__version__)
    print("cuda:", torch.version.cuda)
    print("_GLIBCXX_USE_CXX11_ABI", torch._C._GLIBCXX_USE_CXX11_ABI)

    subprocess.run("pip list | grep tensorrt", shell=True)
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("which nvcc", shell=True)
    subprocess.run("nvcc --version", shell=True)
    # subprocess.run("trtllm-build -h", shell=True)
    # subprocess.run("tritonserver -h", shell=True)

    subprocess.run("nvidia-smi --list-gpus", shell=True, check=True)
    print("--" * 20)
    cmd = f"ldd /opt/tritonserver/backends/python/triton_python_backend_stub"
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)

    print("--" * 20)
    cmd = f"ldd /python_backend/build/triton_python_backend_stub"
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)

    for infer_dir in [
        "whisper",
        "whisper_bls",
        "whisper_infer_bls",
        "whisper_tensorrt_llm_cpprunner",
        "whisper_tensorrt_llm",
    ]:
        cmd = f"cp /python_backend/build/triton_python_backend_stub {model_repo}/{infer_dir}"
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)


"""
# run tritonserver with whisper(python BE)
APP_NAME=whisper TENSORRT_LLM_MODEL_NAME=whisper modal serve src/llm/trtllm/whisper/tritonserver.py

# run tritonserver with whisper_infer_bls + whisper_tensorrt_llm_cpprunner(encoder-decoder python BE)
APP_NAME=whisper TENSORRT_LLM_MODEL_NAME=whisper_infer_bls,whisper_tensorrt_llm_cpprunner modal serve src/llm/trtllm/whisper/tritonserver.py 

# run tritonserver with whisper_bls + whisper_tensorrt_llm(encoder-decoder tensorrtllm BE)
APP_NAME=whisper TENSORRT_LLM_MODEL_NAME=whisper_bls,whisper_tensorrt_llm modal serve src/llm/trtllm/whisper/tritonserver.py 

# curl server is ready
curl -vv -X GET "https://weege009--tritonserver-serve-dev.modal.run/v2/health/ready" -H  "accept: application/json"

# run grpc tritonserver by tcp tunnel and http server
APP_NAME=whisper TENSORRT_LLM_MODEL_NAME=whisper modal run src/llm/trtllm/whisper/tritonserver.py
APP_NAME=whisper TENSORRT_LLM_MODEL_NAME=whisper_bls,whisper_tensorrt_llm modal run src/llm/trtllm/whisper/tritonserver.py 
APP_NAME=whisper TENSORRT_LLM_MODEL_NAME=whisper_infer_bls,whisper_tensorrt_llm_cpprunner modal run src/llm/trtllm/whisper/tritonserver.py 
"""


@app.local_entrypoint()
def main():
    run.remote()
