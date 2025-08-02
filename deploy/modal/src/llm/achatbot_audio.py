import os
import subprocess
import asyncio

import modal

ASR_TAG = os.getenv("ASR_TAG", "whisper_trtllm_asr")

app = modal.App("trtllm-run-whisper-achatbot")
# https://github.com/NVIDIA/TensorRT-LLM/tree/v0.18.0/examples/whisper
# https://github.com/NVIDIA/TensorRT-LLM/blob/v0.20.0/examples/models/core/whisper/README.md
GIT_TAG_OR_HASH = os.getenv("GIT_TAG_OR_HASH", "v0.18.0")
WHISPER_DIRS = {
    "v0.18.0": "/TensorRT-LLM/examples/whisper",
    "v0.20.0": "/TensorRT-LLM/examples/models/core/whisper",
}

trtllm_image = (
    (
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
    )
    .pip_install(
        f"achatbot=={os.getenv('ACHATBOT_VERSION', '0.0.24')}",
        extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
    )
    .env(
        {
            "ACHATBOT_PKG": "1",
            "ASR_TAG": ASR_TAG,
        }
    )
)

TRT_MODEL_DIR = "/root/.achatbot/models"
trt_model_vol = modal.Volume.from_name("triton_trtllm_models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
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
async def run(
    func: callable = None,
    **kwargs,
) -> None:
    import torch

    print("torch:", torch.__version__)
    print("cuda:", torch.version.cuda)
    print("_GLIBCXX_USE_CXX11_ABI", torch._C._GLIBCXX_USE_CXX11_ABI)

    subprocess.run("pip show achatbot", shell=True)
    subprocess.run("pip show tensorrt", shell=True)
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("which nvcc", shell=True)
    subprocess.run("nvcc --version", shell=True)

    print(f"{kwargs=}")
    if asyncio.iscoroutinefunction(func):
        await func(**kwargs)
    else:
        func(**kwargs)


async def transcribe(**kwargs):
    from achatbot.common.interface import IAsr
    from achatbot.modules.speech.asr import ASREnvInit
    from achatbot.common.session import Session
    from achatbot.common.types import SessionCtx
    from achatbot.common.logger import Logger

    Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)

    asr: IAsr = ASREnvInit.initASREngine(os.getenv("ASR_TAG", "whisper_trtllm_asr"), **kwargs)
    audio_file = os.path.join(ASSETS_DIR, "asr_example_zh.wav")
    session = Session(**SessionCtx("test_client_id", 16000, 2).__dict__)
    asr.set_audio_data(audio_file)
    res = await asr.transcribe(session)
    print(res)


"""
# C++ runtime
ASR_TAG=whisper_trtllm_asr modal run src/llm/achatbot_audio.py --task transcribe --no-use-py-session
GIT_TAG_OR_HASH=v0.20.0 ASR_TAG=whisper_trtllm_asr modal run src/llm/achatbot_audio.py --task transcribe --no-use-py-session

ASR_TAG=whisper_trtllm_asr modal run src/llm/achatbot_audio.py --task transcribe --no-use-py-session --engine-dir trt_engines_float16_int8
GIT_TAG_OR_HASH=v0.20.0 ASR_TAG=whisper_trtllm_asr modal run src/llm/achatbot_audio.py --task transcribe --no-use-py-session --engine-dir trt_engines_float16_int8

GIT_TAG_OR_HASH=v0.20.0 ASR_TAG=whisper_trtllm_asr modal run src/llm/achatbot_audio.py --task transcribe --no-use-py-session --engine-dir trt_engines_float16_int4
GIT_TAG_OR_HASH=v0.20.0 ASR_TAG=whisper_trtllm_asr modal run src/llm/achatbot_audio.py --task transcribe --no-use-py-session --engine-dir trt_engines_float16_int4

# Python runtime
ASR_TAG=whisper_trtllm_asr modal run src/llm/achatbot_audio.py --task transcribe --use-py-session
ASR_TAG=whisper_trtllm_asr modal run src/llm/achatbot_audio.py --task transcribe --use-py-session --engine-dir trt_engines_float16_int8
ASR_TAG=whisper_trtllm_asr modal run src/llm/achatbot_audio.py --task transcribe --use-py-session --engine-dir trt_engines_float16_int4
"""


@app.local_entrypoint()
def main(
    task: str = "transcribe",
    engine_dir: str = "trt_engines_float16",
    max_batch_size: int = 1,  # need <= build decoder_model_config.max_batch_size
    use_py_session: bool = True,
):
    tasks = {
        "transcribe": transcribe,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    local_trt_build_dir = os.path.join(TRT_MODEL_DIR, "whisper", engine_dir)
    run.remote(
        func=tasks[task],
        engine_dir=local_trt_build_dir,
        max_batch_size=max_batch_size,
        use_py_session=use_py_session,
    )
