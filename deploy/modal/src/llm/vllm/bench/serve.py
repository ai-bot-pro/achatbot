import os
import modal


vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.7.3",
        "huggingface_hub[hf_transfer]==0.26.2",
        "flashinfer-python==0.2.0.post2",  # pinning, very unstable
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.5",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
)

# PP need close v1
# vllm_image = vllm_image.env({"VLLM_USE_V1": "1"})

MODEL_NAME = os.getenv("MODEL", "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16")


hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
vllm_profile = modal.Volume.from_name("vllm_profile", create_if_missing=True)

PROFILE_DIR = "/root/vllm_profile"

vllm_image = vllm_image.env(
    {
        "VLLM_USE_V1": "1",
        "PROFILER_DIR": PROFILE_DIR if os.getenv("IS_PROFILE") else None,
        "VLLM_RPC_TIMEOUT": "1800000",
    }
)


app = modal.App("vllm-bench")


MINUTES = 60  # seconds

VLLM_PORT = 8000

# NOTE: need to set the image_gpu env variable to L4S or L4S:8 (NUM_GPU)
IMAGE_GPU = os.getenv("IMAGE_GPU", "L40S")
TP_CN = os.getenv("TP", "1")
PP_CN = os.getenv("PP", "1")
EP_CN = os.getenv("EP", "1")

"""
NOTE: 
- need to run the following command ,then to curl the server health check endpoint
- modal serve 
modal serve src/llm/vllm/bench/serve.py
IS_PROFILE=1 modal serve src/llm/vllm/bench/serve.py

curl -X GET "https://weedge--vllm-bench-serve-dev.modal.run/health" -H  "accept: application/json"
"""


@app.function(
    image=vllm_image,
    gpu=IMAGE_GPU,
    # how many requests can one replica handle? tune carefully!
    allow_concurrent_inputs=100,
    # how long should we stay up with no requests?
    container_idle_timeout=15 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        PROFILE_DIR: vllm_profile,
    },
)
@modal.web_server(port=VLLM_PORT, startup_timeout=5 * MINUTES)
def serve():
    import subprocess
    import os

    os.makedirs(PROFILE_DIR, exist_ok=True)
    cmd = []
    if os.getenv("PROFILER_DIR"):
        cmd.append(f"VLLM_TORCH_PROFILER_DIR={os.getenv('PROFILER_DIR')}")

    cmd.extend(
        [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--uvicorn-log-level=info",
            "--model",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--tensor-parallel-size",
            TP_CN,
            "--pipeline-parallel-size",
            PP_CN,
        ]
    )
    print(cmd)

    subprocess.Popen(" ".join(cmd), shell=True)
