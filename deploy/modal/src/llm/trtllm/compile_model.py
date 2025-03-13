import os
import modal

app = modal.App("trtllm")

trtllm_image = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.10",
    )
    .entrypoint([])  # remove verbose logging by base image on entry
    .apt_install(
        "git", "git-lfs", "openmpi-bin", "libopenmpi-dev", "wget"
    )  # OpenMPI for distributed communication
    .pip_install(
        "tensorrt_llm==0.17.0",
        "pynvml<12",  # avoid breaking change to pynvml version API for tensorrt_llm
        pre=True,
        extra_index_url="https://pypi.nvidia.com",
    )
    .env({})
)


HF_MODEL_DIR = "/root/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
TRT_MODEL_DIR = "/root/trt_models"
trt_model_vol = modal.Volume.from_name("triton_trtllm_models", create_if_missing=True)


GIT_TAG_OR_HASH = "v0.17.0"
CONVERSION_SCRIPT_URL = f"https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/{GIT_TAG_OR_HASH}/examples/qwen/convert_checkpoint.py"


@app.function(
    gpu="L4",
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
) -> str:
    import subprocess

    cmd = f"which nvcc".split(" ")
    subprocess.run(cmd, cwd="/", check=True)
    cmd = f"nvcc --version".split(" ")
    subprocess.run(cmd, cwd="/", check=True)

    cmd = f"ls {HF_MODEL_DIR}/{hf_repo_dir}".split(" ")
    subprocess.run(cmd, cwd="/")

    cmd = f"wget {convert_script_url} -O /root/convert.py".split(" ")
    subprocess.run(cmd, cwd="/")

    local_hf_model_vol = os.path.join(HF_MODEL_DIR, hf_repo_dir)
    local_hf_model_vol = os.path.join(TRT_MODEL_DIR, app_name, f"tllm_checkpoint_{trt_dtype}")

    print("「Convert」 Converting checkpoint to TensorRT weights")
    cmd = (
        f"python /root/convert.py --model_dir {local_hf_model_vol} "
        + f"--output_dir {local_hf_model_vol} --dtype {trt_dtype} "
        + f"{convert_other_args}"
    )
    print(cmd)
    subprocess.run(cmd.strip().split(" "), cwd="/")

    print("「Compilation」Building TensorRT engines")
    local_trt_build_dir = os.path.join(TRT_MODEL_DIR, app_name, f"trt_engines_{trt_dtype}")
    cmd = (
        f"trtllm-build --checkpoint_dir {local_hf_model_vol} --output_dir {local_trt_build_dir} "
        + f"--gemm_plugin {trt_dtype} "
        + f"{compile_other_args}"
    )
    print(cmd)
    subprocess.run(cmd.strip().split(" "), cwd="/")


"""
modal run src/llm/trtllm/compile_model.py \
    --app-name "tts-spark" \
    --hf-repo-dir "SparkAudio/Spark-TTS-0.5B/LLM" \
    --trt-dtype "bfloat16" \
    --convert-other-args "" \
    --compile-other-args "--max_batch_size 16 --max_num_tokens 32768"

modal run src/llm/trtllm/compile_model.py \
    --app-name "tts-spark" \
    --hf-repo-dir "SparkAudio/Spark-TTS-0.5B/LLM" \
    --trt-dtype "bfloat16" \
    --convert-script-url "https://raw.githubusercontent.com/SparkAudio/Spark-TTS/refs/heads/main/runtime/triton_trtllm/scripts/convert_checkpoint.py" \
    --convert-other-args "" \
    --compile-other-args "--max_batch_size 16 --max_num_tokens 32768"
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
