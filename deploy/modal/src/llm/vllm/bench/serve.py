import os
import modal

MODEL_NAME = os.getenv("MODEL", "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16")

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

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
vllm_profile = modal.Volume.from_name("vllm_profile", create_if_missing=True)

PROFILE_DIR = "/root/vllm_profile"

# PP need close v1
vllm_image = vllm_image.env(
    {
        "VLLM_USE_V1": os.getenv("VLLM_USE_V1", "1"),
        "TP_CN": os.getenv("TP", "1"),
        "PP_CN": os.getenv("PP", "1"),
        "EP_CN": os.getenv("EP", "1"),
        "PROFILER_DIR": PROFILE_DIR if os.getenv("IS_PROFILE") else None,
        "VLLM_RPC_TIMEOUT": "1800000",
    }
)


app = modal.App("vllm-bench")


MINUTES = 60  # seconds

VLLM_PORT = 8000

"""
NOTE: 
- need to run the following command ,then to curl the server health check endpoint
- modal serve container scalling when concurrent request is high

IMAGE_GPU=L40S modal serve src/llm/vllm/bench/serve.py
TP=1 PP=1 IMAGE_GPU=L40S modal serve src/llm/vllm/bench/serve.py
VLLM_USE_V1=0 TP=2 PP=2 IMAGE_GPU=L4:4 modal serve src/llm/vllm/bench/serve.py
VLLM_USE_V1=0 TP=4 PP=2 IMAGE_GPU=L4:8 modal serve src/llm/vllm/bench/serve.py

# only scale TP
TP=2 PP=1 IMAGE_GPU=L4 modal serve src/llm/vllm/bench/serve.py
TP=4 PP=1 IMAGE_GPU=L4 modal serve src/llm/vllm/bench/serve.py
TP=8 PP=1 IMAGE_GPU=L4 modal serve src/llm/vllm/bench/serve.py

IS_PROFILE=1 modal serve src/llm/vllm/bench/serve.py

curl -vv -X GET "https://weedge--vllm-bench-serve-dev.modal.run/health" -H  "accept: application/json"
"""


@app.function(
    image=vllm_image,
    # NOTE: need to set the image_gpu env variable to L4S or L4S:8 (NUM_GPU)
    gpu=os.getenv("IMAGE_GPU", "L4"),
    # how many requests can one replica handle? tune carefully!
    allow_concurrent_inputs=100,
    # how long should we stay up with no requests?
    scaledown_window=15 * MINUTES,
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
            os.getenv("TP_CN", "1"),
            "--pipeline-parallel-size",
            os.getenv("PP_CN", "1"),
        ]
    )
    print(cmd)

    subprocess.Popen(" ".join(cmd), shell=True)
