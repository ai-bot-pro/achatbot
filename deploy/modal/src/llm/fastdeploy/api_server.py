# author: weedge (weege007@gmail.com)

import os
import subprocess

import modal


LLM_MODEL = os.getenv("LLM_MODEL", "baidu/ERNIE-4.5-0.3B-Paddle")
IMAGE_GPU = os.getenv("IMAGE_GPU", "A100")
FASTDEPLOY_VERSION = os.getenv("FASTDEPLOY_VERSION", "stable")  # stable, nightly
GPU_ARCHS = os.getenv("GPU_ARCHS", "80_90")  # 80_90, 86_89
QUANTIZATION = os.getenv("quantization", "wint4")  # wint8, wint4
TP = os.getenv("TP", "1")

app = modal.App("fastdeploy-api-server")
img = (
    # use openai triton
    # https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/get_started/installation/nvidia_gpu.md
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        # "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy-cuda-12.6:2.0.0",
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "clang", "cmake", "ninja-build")
    .pip_install(
        "paddlepaddle-gpu==3.1.0",
        index_url=" https://www.paddlepaddle.org.cn/packages/stable/cu126/",
    )
    .run_commands(
        f"python -m pip install fastdeploy-gpu -i https://www.paddlepaddle.org.cn/packages/{FASTDEPLOY_VERSION}/fastdeploy-gpu-{GPU_ARCHS}/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
    )
    .env(
        {
            # "TQDM_DISABLE": "1",
            "LLM_MODEL": LLM_MODEL,
            "IMAGE_GPU": IMAGE_GPU,
            "TP": TP,
            "QUANTIZATION": QUANTIZATION,
        }
    )
)

HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)
VIDEO_OUTPUT_DIR = "/gen_video"
video_out_vol = modal.Volume.from_name("gen_video", create_if_missing=True)


@app.function(
    gpu=os.getenv("IMAGE_GPU", "A100"),
    cpu=2.0,
    image=img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_dir,
        VIDEO_OUTPUT_DIR: video_out_vol,
    },
    timeout=86400,  # default 300s
    max_containers=1,
)
@modal.web_server(port=8180, startup_timeout=60 * 60)
def openai():
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    subprocess.run("python -m fastdeploy.entrypoints.openai.api_server --help", shell=True)

    TP = os.getenv("TP")
    QUANTIZATION = os.getenv("QUANTIZATION")
    LLM_MODEL = os.getenv("LLM_MODEL")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, LLM_MODEL)
    cmd = f"""
    python -m fastdeploy.entrypoints.openai.api_server \
        --model {MODEL_PATH} \
        --port 8180 \
        --metrics-port 8181 \
        --engine-worker-queue-port 8182 \
        --max-model-len 32768 \
        --max-num-seqs 32 \
        --reasoning-parser ernie-45-vl \
        --tensor-parallel-size {TP} \
        --quantization {QUANTIZATION} \
        --enable-mm
    """
    print(cmd)
    subprocess.Popen(cmd, env=os.environ, shell=True)


@app.function(
    gpu=os.getenv("IMAGE_GPU", "A100"),
    cpu=2.0,
    image=img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_dir,
        VIDEO_OUTPUT_DIR: video_out_vol,
    },
    timeout=86400,  # default 300s
    max_containers=1,
)
@modal.web_server(port=9904, startup_timeout=60 * 60)
def generate():
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    subprocess.run("python -m fastdeploy.entrypoints.api_server --help", shell=True)

    TP = os.getenv("TP")
    QUANTIZATION = os.getenv("QUANTIZATION")
    LLM_MODEL = os.getenv("LLM_MODEL")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, LLM_MODEL)
    cmd = f"""
    python -m fastdeploy.entrypoints.api_server \
        --model {MODEL_PATH} \
        --port 9904 \
        --max-model-len 32768 \
        --max-num-seqs 32 \
        --reasoning-parser ernie-45-vl \
        --tensor-parallel-size {TP} \
        --quantization {QUANTIZATION} \
        --enable-mm
    """
    print(cmd)
    subprocess.Popen(cmd, env=os.environ, shell=True)


"""
# https://paddlepaddle.github.io/FastDeploy/get_started/quick_start_vl/
# https://paddlepaddle.github.io/FastDeploy/parameters/
# https://paddlepaddle.github.io/FastDeploy/quantization/online_quantization/

# 0. download paddle model
modal run src/download_models.py --repo-ids "baidu/ERNIE-4.5-VL-28B-A3B-Paddle"

# 1. run serve (just to test)
LLM_MODEL=baidu/ERNIE-4.5-VL-28B-A3B-Paddle GPU_ARCHS=86_89 IMAGE_GPU=L40s QUANTIZATION=wint4 TP=1 modal serve src/llm/fastdeploy/api_server.py
LLM_MODEL=baidu/ERNIE-4.5-VL-28B-A3B-Paddle GPU_ARCHS=80_90 IMAGE_GPU=A100-80GB QUANTIZATION=wint8 TP=1 modal serve src/llm/fastdeploy/api_server.py

# 2. init to run and check health
curl -i https://weedge--fastdeploy-api-server-openai-dev.modal.run/health


# 3. run image prompt chat
curl -X POST "https://weedge--fastdeploy-api-server-openai-dev.modal.run/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"}},
      {"type": "text", "text": "What era does this artifact belong to?"}
    ]}
  ],
  "metadata": {"enable_thinking": false}
}'
curl -X POST "https://weedge--fastdeploy-api-server-openai-dev.modal.run/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"}},
      {"type": "text", "text": "What era does this artifact belong to?"}
    ]}
  ],
  "metadata": {"enable_thinking": true}
}'
curl -X POST "https://weedge--fastdeploy-api-server-openai-dev.modal.run/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"}},
      {"type": "text", "text": "What era does this artifact belong to?"}
    ]}
  ],
  "metadata": {"enable_thinking": false},"stream":true
}'
curl -X POST "https://weedge--fastdeploy-api-server-openai-dev.modal.run/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"}},
      {"type": "text", "text": "What era does this artifact belong to?"}
    ]}
  ],
  "metadata": {"enable_thinking": true},"stream":true
}'

# 4. run cli with streaming
modal run src/llm/fastdeploy/api_server.py --url "https://weege126--fastdeploy-api-server-openai-dev.modal.run"
FASTDEPLOY_SERVE_URL https://weege126--fastdeploy-api-server-openai-dev.modal.run python src/llm/fastdeploy/cli.py
"""


@app.local_entrypoint()
def main(url: str = "https://weedge--fastdeploy-api-server-openai-dev.modal.run"):
    import openai

    client = openai.Client(base_url=f"{url}/v1", api_key="null")

    response = client.chat.completions.create(
        model="null",
        messages=[
            {"role": "system", "content": "I'm a helpful AI assistant."},
            {"role": "user", "content": "Rewrite Li Bai's 'Quiet Night Thought' as a modern poem"},
        ],
        stream=True,
    )
    for chunk in response:
        if chunk.choices[0].delta:
            print(chunk.choices[0].delta.content, end="")
    print("\n")
