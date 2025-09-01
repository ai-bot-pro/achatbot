import os
import modal

app = modal.App("trtllm-compile")

GIT_TAG_OR_HASH = "0.18.0"
CONVERSION_SCRIPT_URL = f"https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/v{GIT_TAG_OR_HASH}/examples/llama/convert_checkpoint.py"

trtllm_image = (
    # https://nvidia.github.io/TensorRT-LLM/release-notes.html
    # https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release/tags
    # https://modal.com/docs/examples/trtllm_latency
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",  # TRT-LLM requires Python 3.12
    )
    .entrypoint([])  # remove verbose logging by base image on entry
    .apt_install("openmpi-bin", "libopenmpi-dev", "git", "git-lfs", "wget")
    .pip_install(
        f"tensorrt-llm=={GIT_TAG_OR_HASH}",
        "pynvml<12",  # avoid breaking change to pynvml version API
        "flashinfer-python==0.2.5",
        "cuda-python==12.9.1",
        pre=True,
        extra_index_url="https://pypi.nvidia.com",
    )
    # .env({"TORCH_CUDA_ARCH_LIST": "8.0 8.9 9.0 9.0a"})
)

HF_MODEL_DIR = "/root/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
TRT_MODEL_DIR = "/root/trt_models"
trt_model_vol = modal.Volume.from_name("triton_trtllm_models", create_if_missing=True)


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    retries=0,
    image=trtllm_image,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        TRT_MODEL_DIR: trt_model_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
)
def trtllm_build(
    app_name: str,
    hf_repo_dir: str,
    trt_dtype: str = "bfloat16",
    convert_script_url: str = CONVERSION_SCRIPT_URL,
    convert_other_args: str = "",
    compile_other_args: str = "",
) -> None:
    import subprocess

    cmd = f"pip show tensorrt".split(" ")
    subprocess.run(cmd, cwd="/", check=True)

    cmd = f"which nvcc".split(" ")
    subprocess.run(cmd, cwd="/", check=True)
    cmd = f"nvcc --version".split(" ")
    subprocess.run(cmd, cwd="/", check=True)

    cmd = f"ls {HF_MODEL_DIR}/{hf_repo_dir}".split(" ")
    subprocess.run(cmd, cwd="/", check=True)

    cmd = f"wget {convert_script_url} -O /root/convert.py".split(" ")
    subprocess.run(cmd, cwd="/", check=True)

    local_hf_model_dir = os.path.join(HF_MODEL_DIR, hf_repo_dir)
    local_trt_model_dir = os.path.join(TRT_MODEL_DIR, app_name, f"tllm_checkpoint_{trt_dtype}")

    print("「Convert」 Converting checkpoint to TensorRT weights")
    cmd = (
        f"python /root/convert.py --model_dir {local_hf_model_dir} "
        + f"--output_dir {local_trt_model_dir} --dtype {trt_dtype} "
        + f"{convert_other_args}"
    )
    print(cmd)
    subprocess.run(cmd.strip().split(" "), cwd="/", check=True)

    print("「Compilation」Building TensorRT engines")
    local_trt_build_dir = os.path.join(TRT_MODEL_DIR, app_name, f"trt_engines_{trt_dtype}")
    cmd = (
        f"trtllm-build --checkpoint_dir {local_trt_model_dir} --output_dir {local_trt_build_dir} "
        + f"--gemm_plugin {trt_dtype} "
        + f"{compile_other_args}"
    )
    print(cmd)
    subprocess.run(cmd.strip().split(" "), cwd="/", check=True)

    trt_model_vol.commit()


"""
# - https://github.com/NVIDIA/TensorRT-LLM/blob/v0.18.0/examples/mixtral/README.md
# - https://github.com/NVIDIA/TensorRT-LLM/blob/v0.18.0/examples/quantization/README.md

modal run src/llm/trtllm/translation/compile_model.py \
    --app-name "seed-x" \
    --hf-repo-dir "ByteDance-Seed/Seed-X-PPO-7B" \
    --trt-dtype "bfloat16" \
    --convert-other-args "" \
    --compile-other-args "--max_batch_size 16 --max_num_tokens 16384"

"""


@app.local_entrypoint()
def main(
    app_name: str,
    hf_repo_dir: str,
    trt_dtype: str = "bfloat16",
    convert_script_url: str = CONVERSION_SCRIPT_URL,
    convert_other_args: str = "",
    compile_other_args: str = "",
):
    trtllm_build.remote(
        app_name,
        hf_repo_dir,
        trt_dtype,
        convert_script_url,
        convert_other_args,
        compile_other_args,
    )
