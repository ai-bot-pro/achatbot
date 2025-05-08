import os
import modal

app = modal.App("trtllm-compile-whisper")
# https://github.com/NVIDIA/TensorRT-LLM/blob/v0.20.0rc0/examples/models/core/whisper/README.md
# GIT_TAG_OR_HASH = "v0.20.0rc0"
# https://github.com/NVIDIA/TensorRT-LLM/tree/v0.15.0/examples/whisper
GIT_TAG_OR_HASH = os.getenv("GIT_TAG_OR_HASH", "v0.18.0")
CONVERSION_SCRIPT_URLS = {
    "0.15.0.dev2024110500": "https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/v0.15.0/examples/whisper/convert_checkpoint.py",
    "v0.18.0": "https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/v0.18.0/examples/whisper/convert_checkpoint.py",
    # "v0.20.0rc0": "https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/v0.20.0rc0/examples/models/core/whisper/convert_checkpoint.py",
}


trtllm_image = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git", "clang", "git-lfs", "openmpi-bin", "libopenmpi-dev", "wget"
    )  # OpenMPI for distributed communication
    .pip_install(
        f"tensorrt-llm=={GIT_TAG_OR_HASH}",
        # "pynvml<12",  # avoid breaking change to pynvml version API for tensorrt_llm
        pre=True,
        extra_index_url="https://pypi.nvidia.com",
    )
    # https://github.com/flashinfer-ai/flashinfer/issues/738
    # .pip_install(
    #    "wheel",
    #    "setuptools==75.6.0",
    #    "packaging==23.2",
    #    "ninja==1.11.1.3",
    #    "build==1.2.2.post1",
    # )
    # .run_commands(
    #    "git clone https://github.com/flashinfer-ai/flashinfer.git --recursive",
    #    'cd /flashinfer && git checkout v0.2.5 && TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0a" FLASHINFER_ENABLE_AOT=1 pip install --no-build-isolation --verbose --editable .',
    # )
    .env({})
)


MODEL_DIR = "/root/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
TRT_MODEL_DIR = "/root/trt_models"
trt_model_vol = modal.Volume.from_name("triton_trtllm_models", create_if_missing=True)


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    retries=0,
    image=trtllm_image,
    volumes={
        MODEL_DIR: hf_model_vol,
        TRT_MODEL_DIR: trt_model_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
)
def trtllm_build(
    app_name: str,
    model_name: str,
    trt_dtype: str = "bfloat16",
    convert_script_url: str = CONVERSION_SCRIPT_URLS[GIT_TAG_OR_HASH],
    convert_other_args: str = "",
    compile_other_args: str = "",
) -> str:
    import subprocess
    import torch

    print("torch:", torch.__version__)
    print("cuda:", torch.version.cuda)
    print("_GLIBCXX_USE_CXX11_ABI", torch._C._GLIBCXX_USE_CXX11_ABI)

    subprocess.run("pip show tensorrt", shell=True)
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("which nvcc", shell=True)
    subprocess.run("nvcc --version", shell=True)
    subprocess.run("trtllm-build -h", shell=True)

    cmd = f"wget {convert_script_url} -O /root/convert.py".split(" ")
    subprocess.run(cmd, cwd="/", check=True)

    local_model_dir = MODEL_DIR
    local_trt_model_dir = os.path.join(TRT_MODEL_DIR, app_name, f"tllm_checkpoint_{trt_dtype}")
    if "int8" in convert_other_args:
        local_trt_model_dir += "_int8"
    elif "int4" in convert_other_args:
        local_trt_model_dir += "_int4"

    print("「Convert」Converting checkpoint to TensorRT weights")
    cmd = (
        f"python /root/convert.py --model_dir {local_model_dir} "
        + f"--model_name {model_name} "
        + f"--output_dir {local_trt_model_dir} "
        + f"--dtype {trt_dtype} --logits_dtype {trt_dtype} "
        + f"{convert_other_args}"
    )
    print(cmd)
    subprocess.run(cmd.strip().split(" "), cwd="/", check=True)

    local_trt_build_dir = os.path.join(TRT_MODEL_DIR, app_name, f"trt_engines_{trt_dtype}")
    if "int8" in convert_other_args:
        local_trt_build_dir += "_int8"
    elif "int4" in convert_other_args:
        local_trt_build_dir += "_int4"
    print("「Compilation」Building Whisper Encoder TensorRT engines")
    cmd = (
        f"trtllm-build --checkpoint_dir {local_trt_model_dir}/encoder "
        + f"--output_dir {local_trt_build_dir}/encoder "
        + "--moe_plugin disable "
        + "--gemm_plugin disable "
        + "--max_input_len 3000 --max_seq_len 3000 "
        + "--max_batch_size 8 "
        + f"--bert_attention_plugin {trt_dtype} "
        + f"{compile_other_args}"
    )
    print(cmd)
    subprocess.run(cmd.strip().split(" "), cwd="/", check=True)

    print("「Compilation」Building Whisper Decoder TensorRT engines")
    cmd = (
        f"trtllm-build --checkpoint_dir {local_trt_model_dir}/decoder "
        + f"--output_dir {local_trt_build_dir}/decoder "
        + "--moe_plugin disable "
        + "--max_input_len 14 --max_seq_len 114"  # decoder_input_ids: torch.Size([1, 4])
        + "--max_beam_width 4 "
        + "--max_batch_size 8 "
        + "--max_encoder_input_len 3000 "
        + f"--gemm_plugin {trt_dtype} "
        + f"--bert_attention_plugin {trt_dtype} "
        + f"--gpt_attention_plugin {trt_dtype} "
        + f"{compile_other_args}"
    )
    print(cmd)
    subprocess.run(cmd.strip().split(" "), cwd="/", check=True)


"""
# NOTE: 
# - to use inflight batching and paged kv cache features in C++ runtime, please make sure you have set --paged_kv_cache enable and --remove_input_padding enable (which is by default enabled) in the trtllm-build command. 
# - Meanwhile, if using Python runtime, it is recommended to disable these flag by --paged_kv_cache disable and --remove_input_padding disable to avoid any unnecessary overhead.

# C++ runtime

modal run src/llm/trtllm/whisper/compile_model.py \
    --app-name "whisper" \
    --model-name "large-v3" \
    --trt-dtype "float16" \
    --convert-other-args "" \
    --compile-other-args ""

## use INT8 weight-only Quant
modal run src/llm/trtllm/whisper/compile_model.py \
    --app-name "whisper" \
    --model-name "large-v3" \
    --trt-dtype "float16" \
    --convert-other-args "--use_weight_only --weight_only_precision int8" \
    --compile-other-args ""

## use INT4 weight-only Quant
modal run src/llm/trtllm/whisper/compile_model.py \
    --app-name "whisper" \
    --model-name "large-v3" \
    --trt-dtype "float16" \
    --convert-other-args "--use_weight_only --weight_only_precision int4" \
    --compile-other-args ""

# Python runtime

modal run src/llm/trtllm/whisper/compile_model.py \
    --app-name "whisper" \
    --model-name "large-v3" \
    --trt-dtype "float16" \
    --convert-other-args "" \
    --compile-other-args "--paged_kv_cache disable --remove_input_padding disable"

modal run src/llm/trtllm/whisper/compile_model.py \
    --app-name "whisper" \
    --model-name "large-v3" \
    --trt-dtype "float16" \
    --convert-script-url "https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/refs/tags/v0.19.0rc0/examples/whisper/convert_checkpoint.py" \
    --convert-other-args "" \
    --compile-other-args "--paged_kv_cache disable --remove_input_padding disable"

## use INT8 weight-only Quant
modal run src/llm/trtllm/whisper/compile_model.py \
    --app-name "whisper" \
    --model-name "large-v3" \
    --trt-dtype "float16" \
    --convert-other-args "--use_weight_only --weight_only_precision int8" \
    --compile-other-args "--paged_kv_cache disable --remove_input_padding disable"

## use INT4 weight-only Quant
modal run src/llm/trtllm/whisper/compile_model.py \
    --app-name "whisper" \
    --model-name "large-v3" \
    --trt-dtype "float16" \
    --convert-other-args "--use_weight_only --weight_only_precision int4" \
    --compile-other-args "--paged_kv_cache disable --remove_input_padding disable"
"""


@app.local_entrypoint()
def main(
    app_name: str,
    model_name: str,
    trt_dtype: str = "bfloat16",
    convert_script_url: str = CONVERSION_SCRIPT_URLS[GIT_TAG_OR_HASH],
    convert_other_args: str = "",
    compile_other_args: str = "",
):
    trtllm_build.remote(
        app_name,
        model_name,
        trt_dtype,
        convert_script_url,
        convert_other_args,
        compile_other_args,
    )
