# https://github.com/NVIDIA/TensorRT-LLM/blob/v0.20.0rc0/examples/models/core/whisper/README.md#run

import os
import subprocess
import modal

app = modal.App("trtllm-run-whisper")
# https://github.com/NVIDIA/TensorRT-LLM/blob/v0.20.0rc0/examples/models/core/whisper/README.md
# GIT_TAG_OR_HASH = "v0.20.0rc0"
# https://github.com/NVIDIA/TensorRT-LLM/tree/v0.15.0/examples/whisper
GIT_TAG_OR_HASH = os.getenv("GIT_TAG_OR_HASH", "v0.18.0")
WHISPER_DIRS = {
    "0.15.0.dev2024110500": "/TensorRT-LLM/examples/whisper",
    "v0.18.0": "/TensorRT-LLM/examples/whisper",
    # "v0.20.0rc0": "/TensorRT-LLM/examples/models/core/whisper",
}

trtllm_image = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "ffmpeg", "git", "clang", "git-lfs", "openmpi-bin", "libopenmpi-dev", "wget"
    )  # OpenMPI for distributed communication
    .pip_install(
        f"tensorrt-llm=={GIT_TAG_OR_HASH}",
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
    .run_commands(
        f"git clone https://github.com/NVIDIA/TensorRT-LLM.git -b {GIT_TAG_OR_HASH}",
        f"pip install -r {WHISPER_DIRS[GIT_TAG_OR_HASH]}/requirements.txt",
    )
    .env({})
)


TRT_MODEL_DIR = "/root/trt_models"
trt_model_vol = modal.Volume.from_name("triton_trtllm_models", create_if_missing=True)
ASSETS_DIR = "/root/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    retries=0,
    image=trtllm_image,
    volumes={
        TRT_MODEL_DIR: trt_model_vol,
        ASSETS_DIR: assets_dir,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
)
def run(
    app_name: str,
    engine_dir: str = "trt_engines_float16",
    other_args: str = "",
    func: callable = None,
) -> None:
    import torch

    print("torch:", torch.__version__)
    print("cuda:", torch.version.cuda)
    print("_GLIBCXX_USE_CXX11_ABI", torch._C._GLIBCXX_USE_CXX11_ABI)

    subprocess.run("pip show tensorrt", shell=True)
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("which nvcc", shell=True)
    subprocess.run("nvcc --version", shell=True)

    if func is None:
        run_single_wav_test(app_name, engine_dir, other_args)
    else:
        func(app_name, engine_dir, other_args)


def run_single_wav_test(app_name: str, engine_dir: str, other_args: str) -> None:
    local_trt_build_dir = os.path.join(TRT_MODEL_DIR, app_name, engine_dir)
    input_file = os.path.join(ASSETS_DIR, "1221-135766-0002.wav")
    cmd = f"python3 run.py --name single_wav_test --engine_dir {local_trt_build_dir} --input_file {input_file} --assets_dir {ASSETS_DIR} --results_dir {ASSETS_DIR} {other_args}"
    print(cmd)
    try:
        subprocess.run(
            cmd,
            shell=True,
            check=True,
            cwd=WHISPER_DIRS[GIT_TAG_OR_HASH],
        )
    except subprocess.CalledProcessError as e:
        print(f"run stderr: {e.stderr}")


def run_dataset_bench(app_name: str, engine_dir: str, other_args: str) -> None:
    local_trt_build_dir = os.path.join(TRT_MODEL_DIR, app_name, engine_dir)
    cmd = f"python3 run.py --name librispeech_dummy_large_v3 --engine_dir {local_trt_build_dir} --dataset hf-internal-testing/librispeech_asr_dummy --assets_dir {ASSETS_DIR} --results_dir {ASSETS_DIR} --enable_warmup {other_args}"
    print(cmd)
    try:
        subprocess.run(
            cmd,
            shell=True,
            check=True,
            cwd=WHISPER_DIRS[GIT_TAG_OR_HASH],
        )
    except subprocess.CalledProcessError as e:
        print(f"run stderr: {e.stderr}")


"""
# run_single_wav_test, NOTE: no WER eval
## C++ runtime
modal run src/llm/trtllm/whisper/run.py \
    --app-name "whisper" \
    --engine-dir "trt_engines_float16" \
    --other-args "--log_level info"

modal run src/llm/trtllm/whisper/run.py \
    --app-name "whisper" \
    --engine-dir "trt_engines_float16_int8" \
    --other-args "--log_level info"

modal run src/llm/trtllm/whisper/run.py \
    --app-name "whisper" \
    --engine-dir "trt_engines_float16_int4" \
    --other-args "--log_level info"

## Python runtime
modal run src/llm/trtllm/whisper/run.py \
    --other-args "--use_py_session" \
    --app-name "whisper" \
    --engine-dir "trt_engines_float16" \
    --other-args "--log_level info"

# run_dataset_bench, NOTE: have WER eval
## C++ runtime
modal run src/llm/trtllm/whisper/run.py \
    --task "run_dataset_bench" \
    --app-name "whisper" \
    --engine-dir "trt_engines_float16" \
    --other-args "--log_level info"

modal run src/llm/trtllm/whisper/run.py \
    --task "run_dataset_bench" \
    --app-name "whisper" \
    --engine-dir "trt_engines_float16_int8" \
    --other-args "--log_level info"

modal run src/llm/trtllm/whisper/run.py \
    --task "run_dataset_bench" \
    --app-name "whisper" \
    --engine-dir "trt_engines_float16_int4" \
    --other-args "--log_level info"

## Python runtime
modal run src/llm/trtllm/whisper/run.py \
    --task "run_dataset_bench" \
    --app-name "whisper" \
    --engine-dir "trt_engines_float16" \
    --other-args "--log_level info --use_py_session"
"""


@app.local_entrypoint()
def main(
    app_name: str = "whisper",
    engine_dir: str = "trt_engines_float16",
    other_args: str = "",
    task: str = "run_single_wav_test",
):
    tasks = {
        "run_single_wav_test": run_single_wav_test,
        "run_dataset_bench": run_dataset_bench,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(
        app_name,
        engine_dir,
        other_args,
        func=tasks[task],
    )
