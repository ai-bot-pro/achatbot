import os
import sys
import subprocess


import modal


app = modal.App("openai_gpt_oss_evalscope")
IMAGE_GPU = os.getenv("IMAGE_GPU", None)
img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvcr.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs")
    .run_commands(
        "git clone https://github.com/modelscope/evalscope.git",
        "cd /evalscope && git checkout d3bec2b94a81eec2d1cc15af2a29547a4f5c5368",
        "cd /evalscope && pip install -e '.[perf]'",
    )
    .env(
        {
            # "TQDM_DISABLE": "1",
            "LLM_MODEL": os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
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
    eval_out_dir = os.path.join(EVAL_OUTPUT_DIR, "evalscope", MODEL_PATH.split("/")[-1])
    os.makedirs(eval_out_dir, exist_ok=True)

@app.function(
    gpu=IMAGE_GPU,
    cpu=8.0,
    retries=0,
    image=img,
    secrets=[modal.Secret.from_name("achatbot")],
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        EVAL_OUTPUT_DIR: eval_out_vol,
    },
    timeout=86400,  # default 300s
    max_containers=1,
)
def run(func, **kwargs):
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    subprocess.run("which evalscope", shell=True)

    if torch.cuda.is_available():
        gpu_prop = torch.cuda.get_device_properties("cuda")
        print(gpu_prop)

    func(**kwargs)


def run_cmd(cmd, capture_output=False):
    print(cmd)

    os.environ["PYTHONPATH"] = "/evalscope:" + os.environ["PYTHONPATH"]
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


def inference_speed_test(**kwargs):
    capture_output = kwargs.get("capture_output", False)
    url = kwargs.get("url", "http://127.0.0.1:8801/v1/completions")
    extra_args = """'{"ignore_eos": true}'"""

    cmd = f"""evalscope perf \\
        --parallel 1 10 50 100 \\
        --number 5 20 100 200 \\
        --model {model_name} \\
        --outputs-dir {eval_out_dir} \\
        --url {url} \\
        --api openai \\
        --dataset random \\
        --max-tokens 1024 \\
        --min-tokens 1024 \\
        --prefix-length 0 \\
        --min-prompt-length 1024 \\
        --max-prompt-length 1024 \\
        --log-every-n-query 20 \\
        --tokenizer-path {model_path} \\
        --extra-args {extra_args}
    """
    res: subprocess.CompletedProcess = run_cmd(cmd, capture_output=capture_output)
    print(res)


def benchmark_aime25(**kwargs):
    from evalscope.constants import EvalType
    from evalscope import TaskConfig, run_task

    api_url = kwargs.get("url", "http://127.0.0.1:8801/v1")
    task_cfg = TaskConfig(
        model=model_path,  # Model name
        api_url=api_url,  # Model service address
        eval_type=EvalType.SERVICE,  # Evaluation type, here using service evaluation
        datasets=["aime25"],  # Dataset to test
        generation_config={
            "extra_body": {
                "reasoning_effort": "high"
            }  # Model generation parameters, set to high reasoning level
        },
        eval_batch_size=10,  # Concurrent batch size
        timeout=60000,  # Timeout in seconds
    )

    run_task(task_cfg=task_cfg)


"""
- https://github.com/modelscope/evalscope
- https://evalscope.readthedocs.io/en/latest/index.html
- https://evalscope.readthedocs.io/en/latest/best_practice/gpt_oss.html

# task datasets
- https://evalscope.readthedocs.io/en/latest/get_started/supported_dataset/llm.html
# vlm, RAG eval
- https://evalscope.readthedocs.io/en/latest/user_guides/backend/index.html
# api stress test for openai api format
- https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/index.html
# AIGC for text to image
- https://evalscope.readthedocs.io/en/latest/user_guides/aigc/t2i.html
# Arena
- https://evalscope.readthedocs.io/en/latest/user_guides/arena.html

# 3rd eval tools
- τ-bench (dynamic dialogue agent scenarios): https://evalscope.readthedocs.io/en/latest/third_party/tau_bench.html
- BFCL-v3 (function-calling capabilities): https://evalscope.readthedocs.io/en/latest/third_party/bfcl_v3.html 
- Needle in a Haystack( relevant information within a text filled with a large amount of irrelevant data): https://evalscope.readthedocs.io/en/latest/third_party/needle_haystack.html
- ToolBench (Agents): https://evalscope.readthedocs.io/en/latest/third_party/toolbench.html
- LongWriter (measuring the long output quality as well as the output length): https://evalscope.readthedocs.io/en/latest/third_party/longwriter.html


# download model
modal run src/download_models.py --repo-ids "openai/gpt-oss-20b"

# after run vllm serve, then to run evalscope perf
## run inference speed test
modal run src/eval/evalscope.py --task inference_speed_test --url https://weedge--openai-gpt-oss-serve-dev.modal.run/v1/completions

## run AIME25 eval task
modal run src/eval/evalscope.py --task benchmark_aime25 --url https://weedge--openai-gpt-oss-serve-dev.modal.run/v1
"""


@app.local_entrypoint()
def main(
    task: str = "inference_speed_test",
    url: str = "http://127.0.0.1:8801/v1/completions",
    capture_output: bool = False,
):
    print(task)
    run_tasks = {
        "inference_speed_test": inference_speed_test,
        "benchmark_aime25": benchmark_aime25,
    }
    if task not in run_tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(
        run_tasks[task],
        url=url,
        capture_output=capture_output,
    )
