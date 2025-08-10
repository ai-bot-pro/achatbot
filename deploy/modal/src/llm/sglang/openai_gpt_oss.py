import os
import sys
import subprocess
import time


import modal
import urllib

app = modal.App("openai_gpt_oss_sglang")
RUN_IMAGE_GPU = os.getenv("RUN_IMAGE_GPU", None)
SERVE_IMAGE_GPU = os.getenv("SERVE_IMAGE_GPU", None)
RUN_MAX_CONTAINERS = int(os.getenv("RUN_MAX_CONTAINERS", 1))
SERVE_MAX_CONTAINERS = int(os.getenv("SERVE_MAX_CONTAINERS", 1))
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
TP = os.getenv("TP", "1")
SERVE_ARGS = os.getenv("SERVE_ARGS", "")

img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvcr.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs")
    .run_commands(
        "git clone https://github.com/sgl-project/sglang",
        "cd /sglang && git checkout 3817a37d87619469bd5f2fc5d62c20caaedd666a",
        "cd /sglang && pip install -e python[all]",  # make sure you have the correct transformers version installed!
    )
    .run_commands(
        "pip3 install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129"
    )
    .run_commands(
        "pip3 install https://github.com/sgl-project/whl/releases/download/v0.3.3/sgl_kernel-0.3.3-cp39-abi3-manylinux2014_x86_64.whl --force-reinstall"
    )
    .apt_install("libnuma-dev")  # Add NUMA library for sgl_kernel
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCH_CUDA_ARCH_LIST": "8.0 8.9 9.0+PTX 10.0+PTX",
            # "TQDM_DISABLE": "1",
            "LLM_MODEL": LLM_MODEL,
            "TP": TP,
            "SERVE_ARGS": SERVE_ARGS,
        }
    )
)


HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
EVAL_OUTPUT_DIR = "/eval_output"
eval_out_vol = modal.Volume.from_name("eval_output", create_if_missing=True)


with img.imports():
    import torch

    MODEL_PATH = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    model_path = os.path.join(HF_MODEL_DIR, MODEL_PATH)
    model_name = MODEL_PATH.split("/")[-1]
    eval_out_dir = os.path.join(EVAL_OUTPUT_DIR, "sglang", MODEL_PATH.split("/")[-1])
    os.makedirs(eval_out_dir, exist_ok=True)


@app.function(
    gpu=RUN_IMAGE_GPU,
    cpu=8.0,
    retries=0,
    image=img,
    secrets=[modal.Secret.from_name("achatbot")],
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        EVAL_OUTPUT_DIR: eval_out_vol,
    },
    timeout=86400,  # default 300s
    max_containers=RUN_MAX_CONTAINERS,
)
def remote_run(func, **kwargs):
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    if torch.cuda.is_available():
        gpu_prop = torch.cuda.get_device_properties("cuda")
        print(gpu_prop)

    run(func, **kwargs)


def run(func, **kwargs):
    func(**kwargs)


def run_cmd(cmd, capture_output=False):
    print(cmd)

    # os.environ["PYTHONPATH"] = "/sglang:" + os.environ["PYTHONPATH"]
    try:
        res = subprocess.run(
            cmd.strip(),
            shell=True,
            check=True,
            env=os.environ,
            capture_output=capture_output,
        )

    except subprocess.CalledProcessError as e:
        print("erro_code:", e.returncode)
        print("error:", e.stderr)
        raise

    return res


def benchmark(**kwargs):
    """
    https://github.com/sgl-project/sglang/blob/main/python/sglang/bench_serving.py
    """
    cmd = """
    python3 -m sglang.bench_serving --help     
    """
    run_cmd(cmd)

    url = serve.get_web_url()
    print(url)

    test_timeout = kwargs.get("test_timeout", 30 * 60)
    up, start, delay = False, time.time(), 10
    while not up:
        try:
            with urllib.request.urlopen(url + "/health") as response:
                up = response.getcode() == 200
        except Exception:
            if time.time() - start > test_timeout:
                break
            time.sleep(delay)

    assert up, f"Failed health check for server at {url}"

    print(f"Successful health check for server at {url}")

    num_prompts = kwargs.get("num_prompts", 5)
    max_concurrency = kwargs.get("max_concurrency", 1)
    local_time = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d_%H:%M:%S", local_time)
    cmd = f"""
    python3 -m sglang.bench_serving --backend sglang-oai --base-url {url} \\
        --dataset-name random --random-input-len 512 --random-output-len 1024 --random-range-ratio 1 \\
        --num-prompts {num_prompts} --max-concurrency {max_concurrency} \\
        --output-file {eval_out_dir}/res_{formatted_time}.jsonl
    """
    run_cmd(cmd)


def generate(**kwargs):
    pass


@app.function(
    gpu=SERVE_IMAGE_GPU,
    image=img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        EVAL_OUTPUT_DIR: eval_out_vol,
    },
    # secrets=[modal.Secret.from_name("achatbot")],
    timeout=86400,  # default 300s
    max_containers=SERVE_MAX_CONTAINERS,
)
@modal.web_server(port=30000, startup_timeout=60 * 60)
@modal.concurrent(max_inputs=100, target_inputs=4)
def serve():
    """
    modal + sglang :
    - https://modal.com/docs/examples/sgl_vlm
    - https://modal.com/llm-almanac/advisor
    """
    run_cmd("python3 -m sglang.launch_server --help")
    cmd = f"""
    python3 -m sglang.launch_server --model {model_path} --host 0.0.0.0 --port 30000 \\
        --tp {os.getenv("TP", "1")} {os.getenv("SERVE_ARGS", "")}
    """
    print(cmd)
    subprocess.Popen(cmd, shell=True, env=os.environ)


def local_api_completions(**kwargs):
    from openai import OpenAI

    url = serve.get_web_url()
    print(url)

    client = OpenAI(base_url=f"{url}/v1", api_key="EMPTY")

    result = client.chat.completions.create(
        model=MODEL_PATH,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain what MXFP4 quantization is."},
        ],
    )
    print(result)
    print(f"{result.choices[0].message.content=}")
    print(f"{result.choices[0].message.reasoning_content=}")
    print(result.usage)


def local_api_tool_completions(**kwargs):
    from openai import OpenAI

    url = serve.get_web_url()
    print(url)

    client = OpenAI(base_url=f"{url}/v1", api_key="EMPTY")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather in a given city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]

    result = client.chat.completions.create(
        model=MODEL_PATH,
        messages=[{"role": "user", "content": "What's the weather in Berlin right now?"}],
        tools=tools,
    )
    print(result)
    print(f"{result.choices[0].message.content=}")
    print(f"{result.choices[0].message.reasoning_content=}")
    print(result.usage)


@app.local_entrypoint()
def url_request(test_timeout=30 * 60):
    import json
    import time
    import urllib

    url = serve.get_web_url()
    print(f"Running health check for server at {url}")
    print("Note: startup takes a while on the first two iterations, but is much faster after that.")
    print("On the first iteration with a new model, weights are downloaded at ~100 MB/s.")
    print("On the second iteration, a file read profile is recorded and used for future runs.")
    up, start, delay = False, time.time(), 10
    while not up:
        try:
            with urllib.request.urlopen(url + "/health") as response:
                up = response.getcode() == 200
        except Exception:
            if time.time() - start > test_timeout:
                break
            time.sleep(delay)

    assert up, f"Failed health check for server at {url}"

    print(f"Successful health check for server at {url}")

    messages = [{"role": "user", "content": "Testing! Is this thing on?"}]
    print(f"Sending a sample message to {url}", *messages, sep="\n")

    headers = {"Content-Type": "application/json"}
    payload = json.dumps({"messages": messages, "model": MODEL_PATH, "max_tokens": 128})
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=payload.encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req) as response:
        print(json.loads(response.read().decode()))


"""
# 0. download model weight(safetensors), tokenizer and config
modal run src/download_models.py --repo-ids "lmsys/gpt-oss-20b-bf16"
modal run src/download_models.py --repo-ids "lmsys/gpt-oss-120b-bf16"
modal run src/download_models.py --repo-ids "openai/gpt-oss-20b"
modal run src/download_models.py --repo-ids "openai/gpt-oss-120b"


# 1. run server and test with urllib request raw http api
# fp8/bf16
LLM_MODEL=lmsys/gpt-oss-20b-bf16 SERVE_IMAGE_GPU=A100-80GB TP=1 modal run src/llm/sglang/openai_gpt_oss.py::url_request
# mxfp4 
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=A100-80GB TP=1 modal run src/llm/sglang/openai_gpt_oss.py::url_request


# 2. run server and test with openai client sdk
# fp8/bf16
LLM_MODEL=lmsys/gpt-oss-20b-bf16 SERVE_IMAGE_GPU=A100-80GB TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task local_api_completions
LLM_MODEL=lmsys/gpt-oss-20b-bf16 SERVE_IMAGE_GPU=H100 TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task local_api_completions
LLM_MODEL=lmsys/gpt-oss-20b-bf16 SERVE_IMAGE_GPU=L40s:2 TP=2 SERVE_ARGS="--cuda-graph-max-bs 4" modal run src/llm/sglang/openai_gpt_oss.py::main --task local_api_completionsa 
# mxfp4 
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=A100-80GB TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task local_api_tool_completions
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=H100 TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task local_api_tool_completions
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=L40s:2 TP=2 SERVE_ARGS="--cuda-graph-max-bs 4" modal run src/llm/sglang/openai_gpt_oss.py::main --task local_api_completionsa 


# 3. benchmark
LLM_MODEL=lmsys/gpt-oss-20b-bf16 SERVE_IMAGE_GPU=L40s:2 TP=2 SERVE_ARGS="--cuda-graph-max-bs 4" modal run src/llm/sglang/openai_gpt_oss.py::main --task benchmark
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=L40s:2 TP=2 SERVE_ARGS="--cuda-graph-max-bs 4" modal run src/llm/sglang/openai_gpt_oss.py::main --task benchmark
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=A100-80GB TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task benchmark
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=A100-80GB TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task benchmark --num-prompts 20 --max-concurrency 4
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=H100 TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task benchmark
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=H100 TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task benchmark --num-prompts 20 --max-concurrency 4
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=H200 TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task benchmark
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=H200 TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task benchmark --num-prompts 20 --max-concurrency 4
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=H200 TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task benchmark --num-prompts 80 --max-concurrency 16
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=H200 TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task benchmark --num-prompts 160 --max-concurrency 32
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=B200 TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task benchmark
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=B200 TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task benchmark --num-prompts 20 --max-concurrency 4
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=B200 TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task benchmark --num-prompts 80 --max-concurrency 16
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=B200 TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task benchmark --num-prompts 160 --max-concurrency 32

# 4. run server or deploy
# fp8/bf16
LLM_MODEL=lmsys/gpt-oss-20b-bf16 SERVE_IMAGE_GPU=H100 TP=1 modal serve src/llm/sglang/openai_gpt_oss.py
LLM_MODEL=lmsys/gpt-oss-20b-bf16 SERVE_IMAGE_GPU=L40s:2 TP=2 SERVE_ARGS="--cuda-graph-max-bs 4" modal serve src/llm/sglang/openai_gpt_oss.py
# mxfp4
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=H100 TP=1 modal serve src/llm/sglang/openai_gpt_oss.py
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=L40s:2 TP=2 SERVE_ARGS="--cuda-graph-max-bs 4" modal serve src/llm/sglang/openai_gpt_oss.py

"""


@app.local_entrypoint()
def main(
    task: str = "generate",
    reasoning: str = "medium",  # low medium high
    model_identity: str = "You are ChatGPT, a large language model trained by OpenAI.",
    prompt: str = "什么是快乐星球?",
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 128,
    # for chat_tool_stream
    build_in_tool: str = "browser",  # build-in tools: browser,python
    is_apply_patch: bool = False,  # Make apply_patch tool available to the model (default: False)
    show_browser_results: bool = False,  # Show browser results (default: False)
    developer_message: str = "",  # Developer message (default: )
    raw: bool = False,  # Raw mode (does not render Harmony encoding) (default: False)
    # local completions api
    stream: bool = False,
    # benchmark
    num_prompts: int = 5,
    max_concurrency: int = 1,
):
    print(task)
    tasks = {
        "local_api_completions": local_api_completions,
        "local_api_tool_completions": local_api_tool_completions,
        "generate": generate,
        "benchmark": benchmark,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    if "local" in task:
        func = run
    else:
        func = remote_run.remote

    func(
        tasks[task],
        reasoning=reasoning.lower(),
        model_identity=model_identity,
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        build_in_tool=build_in_tool,
        is_apply_patch=is_apply_patch,
        show_browser_results=show_browser_results,
        developer_message=developer_message,
        raw=raw,
        stream=stream,
        num_prompts=num_prompts,
        max_concurrency=max_concurrency,
    )
