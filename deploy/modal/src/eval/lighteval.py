import os
import sys
import subprocess


import modal


app = modal.App("openai_gpt_oss_lighteval")
IMAGE_GPU = os.getenv("IMAGE_GPU", None)
img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvcr.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs")
    .run_commands(
        "git clone https://github.com/huggingface/lighteval",
        "cd /lighteval && git checkout 7693a0fdb24e8b41ec51e3ea7063e6c1a3ec15a6",
        "cd /lighteval && pip install -e .[dev]",  # make sure you have the correct transformers version installed!
    )
    .run_commands(
        # Install  kernels to load flash-attention3 kenerl
        "pip install -U kernels",
    )
    .run_commands(
        # Install  the Triton kernels for MXFP4 compatibility
        "pip install -U git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels",
    )
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            # "TQDM_DISABLE": "1",
            "LLM_MODEL": os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
        }
    )
)

# for transformers==4.55.0 have some issue(cuda arch < 9.0 don't support mxpf4)
if IMAGE_GPU in ["H100", "B100", "H200", "B200"]:
    img = img.pip_install("triton==3.4.0")
else:
    img = img.pip_install("triton==3.3.1")


HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
EVAL_OUTPUT_DIR = "/eval_output"
eval_out_vol = modal.Volume.from_name("eval_output", create_if_missing=True)


with img.imports():
    import torch

    MODEL_PATH = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    model_path = os.path.join(HF_MODEL_DIR, MODEL_PATH)
    eval_out_dir = os.path.join(EVAL_OUTPUT_DIR, "lighteval", MODEL_PATH.split("/")[-1])
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

    if torch.cuda.is_available():
        gpu_prop = torch.cuda.get_device_properties("cuda")
        print(gpu_prop)

    func(**kwargs)


def run_cmd(cmd, capture_output=False):
    print(cmd)

    os.environ["PYTHONPATH"] = "/lighteval/src:" + os.environ["PYTHONPATH"]
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


def tasks(**kwargs):
    capture_output = kwargs.get("capture_output", False)
    res: subprocess.CompletedProcess = run_cmd(
        "lighteval tasks list", capture_output=capture_output
    )
    print(res)


def gpt2_truthfulqa_eval(**kwargs):
    cmd = f"""lighteval accelerate "model_name={model_path}" "leaderboard|truthfulqa:mc|0|0" --save-details --output-dir "{eval_out_dir}" """
    run_cmd(cmd)


def gpt_oss_ifeval_aime25_eval(**kwargs):
    """
    - ifeval: https://github.com/huggingface/lighteval/tree/main/src/lighteval/tasks/extended/ifeval
    -
    """
    reasoniing_tags = kwargs.get(
        "reasoniing_tags",
        "[('<|channel|>analysis<|message|>','<|end|><|start|>assistant<|channel|>final<|message|>')]",
    )
    generation_parameters = kwargs.get(
        "generation_parameters",
        "{temperature:1,top_p:1,top_k:40,min_p:0,max_new_tokens:16384}",
    )
    mode = kwargs.get("mode", "accelerate")

    cmd = f"""lighteval {mode} \\
        model_name={model_path},max_length=16384,skip_special_tokens=False,generation_parameters={generation_parameters} \\
        "extended|ifeval|0|0,lighteval|aime25|0|0" \\
        --save-details \\
        --output-dir "{eval_out_dir}" \\
        --remove-reasoning-tags \\
        --reasoning-tags="{reasoniing_tags}"
    """

    run_cmd(cmd)


"""
- https://github.com/huggingface/lighteval
- https://huggingface.co/docs/lighteval/quicktour

# tasks
task format:
{suite}|{task}|{num_few_shot}|{0 for strict `num_few_shots`, or 1 to allow a truncation if context size is too small}
- https://huggingface.co/docs/lighteval/available-tasks
- https://huggingface.co/docs/lighteval/adding-a-custom-task

# metrics
- https://huggingface.co/docs/lighteval/metric-list
- https://huggingface.co/docs/lighteval/adding-a-new-metric



# download model
modal run src/download_models.py --repo-ids "gpt2"
modal run src/download_models.py --repo-ids "openai/gpt-oss-20b"

# list tasks
modal run src/eval/lighteval.py --task tasks


# run gpt2 eval
IMAGE_GPU=T4 LLM_MODEL=gpt2 modal run src/eval/lighteval.py --task gpt2_truthfulqa_eval

# run gpt-oss eval
## use L40s GPU to inference with bf16
# largest batch_size:2 | Greedy generation:  23%|██▎       | 61/271 [1:07:38<4:22:57, 75.13s/it] | (47GB/48GB, utilization:92%)
IMAGE_GPU=L40s modal run src/eval/lighteval.py --task gpt_oss_ifeval_aime25_eval 

## use H100 GPU to inference with mxfp4
# largest batch_size: 16 | Greedy generation:   9%|▉         | 3/34 [03:36<36:49, 71.26s/it] | (77GB/80GB, utilization:40%)
IMAGE_GPU=H100 modal run src/eval/lighteval.py --task gpt_oss_ifeval_aime25_eval 
"""


@app.local_entrypoint()
def main(
    task: str = "tasks",
    mode: str = "accelerate",
    generation_parameters: str = "{temperature:1,top_p:1,top_k:40,min_p:0,max_new_tokens:16384}",
    reasoniing_tags: str = "[('<|channel|>analysis<|message|>','<|end|><|start|>assistant<|channel|>final<|message|>')]",
    capture_output: bool = False,
):
    print(task)
    run_tasks = {
        "tasks": tasks,
        "gpt2_truthfulqa_eval": gpt2_truthfulqa_eval,
        "gpt_oss_ifeval_aime25_eval": gpt_oss_ifeval_aime25_eval,
    }
    if task not in run_tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(
        run_tasks[task],
        mode=mode,
        generation_parameters=generation_parameters,
        reasoniing_tags=reasoniing_tags,
        capture_output=capture_output,
    )
