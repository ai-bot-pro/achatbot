import os
import modal

app = modal.App("decoupled")

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
        "mkdir -p models/bls_decoupled_sync/1",
        "mkdir -p models/bls_decoupled_async/1",
        "mkdir -p /models/repeat_int32/1",
        "mkdir -p /models/square_int32/1",
        "cp /python_backend/examples/bls_decoupled/sync_model.py models/bls_decoupled_sync/1/model.py",
        "cp /python_backend/examples/bls_decoupled/sync_config.pbtxt models/bls_decoupled_sync/config.pbtxt",
        "cp /python_backend/build/triton_python_backend_stub /models/bls_decoupled_sync",
        "cp /python_backend/examples/bls_decoupled/async_model.py models/bls_decoupled_async/1/model.py",
        "cp /python_backend/examples/bls_decoupled/async_config.pbtxt models/bls_decoupled_async/config.pbtxt",
        "cp /python_backend/build/triton_python_backend_stub /models/bls_decoupled_async",
        "cp /python_backend/examples/decoupled/repeat_model.py /models/repeat_int32/1/model.py",
        "cp /python_backend/examples/decoupled/repeat_config.pbtxt /models/repeat_int32/config.pbtxt",
        "cp /python_backend/build/triton_python_backend_stub /models/repeat_int32",
        "cp /python_backend/examples/decoupled/square_model.py /models/square_int32/1/model.py",
        "cp /python_backend/examples/decoupled/square_config.pbtxt /models/square_int32/config.pbtxt",
        "cp /python_backend/build/triton_python_backend_stub /models/square_int32",
    )
    .pip_install(
        "numpy",
    )
    .env(
        {
            # "PYTHONPATH": os.getenv("IMAGE_PYTHONPATH", ""),
            # "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES", "0"),
            "PYTHONIOENCODING": "utf-8",
            # "APP_NAME": os.getenv("APP_NAME", "tts-spark"),
        }
    )
)


# ⭐️ https://github.com/triton-inference-server/python_backend/blob/main/README.md
@app.function(
    # gpu=os.getenv("IMAGE_GPU", "L4"),
    cpu=2.0,
    image=tritonserver_image,
    volumes={},
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

    cmd = f"tree /models"
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)

    with modal.forward(8001, unencrypted=True) as tunnel:
        print(
            f"use tunnel.tcp_socket = {tunnel.tcp_socket[0]}:{tunnel.tcp_socket[1]} to connect tritonserver with tcp(grpc)"
        )
        cmd = f"tritonserver --model-repository /models --log-verbose 0 --log-info True"
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)


# ⭐️ https://github.com/triton-inference-server/python_backend/blob/main/README.md
@app.function(
    # gpu=os.getenv("IMAGE_GPU", "L4"),
    cpu=2.0,
    image=tritonserver_image,
    volumes={},
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=100,
)
@modal.web_server(port=8000, startup_timeout=5 * 60)
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

    cmd = f"tree /models"
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)

    cmd = f"tritonserver --model-repository /models"
    print(cmd)
    subprocess.Popen(cmd, env=os.environ, shell=True)


"""
# run http tritonserver
modal serve src/llm/trtllm/decoupled/tritonserver.py 

# curl http server is ready
curl -vv -X GET "https://weedge--decoupled-serve-dev.modal.run/v2/health/ready" -H  "accept: application/json"

# run grpc tritonserver by tcp tunnel and http server
modal run src/llm/trtllm/decoupled/tritonserver.py 
"""


@app.local_entrypoint()
def main():
    run.remote()
