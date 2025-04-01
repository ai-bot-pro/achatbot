import os
from time import perf_counter
import modal


app = modal.App("tts-grpc")

achatbot_version = os.getenv("ACHATBOT_VERSION", "0.0.9.post3")

tts_grpc_image = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git",
        "git-lfs",
        "ffmpeg",
        "openmpi-bin",
        "libopenmpi-dev",
        "cmake",
    )  # OpenMPI for distributed communication
    .pip_install(
        f"achatbot[{os.getenv('TTS_TAG', 'tts_generator_spark')}]=={achatbot_version}",
        extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
    )
    .env(
        {
            "ACHATBOT_PKG": "1",
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "info"),
            "TORCH_CUDA_ARCH_LIST": "7.5 8.0 8.6 8.7 8.9 9.0",
            "TTS_TAG": os.getenv("TTS_TAG", "tts_generator_spark"),
            "TTS_TEXT": os.getenv(
                "TTS_TEXT",
                "hello,你好，我是机器人。|万物之始,大道至简,衍化至繁。|君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。",
            ),
            "TTS_MODEL_DIR": "/root/.achatbot/models/SparkAudio/Spark-TTS-0.5B",
            "TTS_LM_TOKENIZER_DIR": "/root/.achatbot/models/SparkAudio/Spark-TTS-0.5B/LLM",
            "LLM_MODEL_NAME_OR_PATH": "/root/.achatbot/models/SparkAudio/Spark-TTS-0.5B/LLM",
            "LLM_TORCH_DTYPE": "bfloat16",
        }
    )
    .pip_install(
        f"achatbot[librosa,soundfile]=={achatbot_version}",
        extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
    )
    # .pip_install("numpy==1.26.4", "transformers==4.48.3")
)

generator = os.getenv("GENERATOR_ENGINE", "transformers")
if generator == "transformers":
    tts_grpc_image = tts_grpc_image.pip_install(
        f"achatbot[transformers]=={achatbot_version}",
    ).env(
        {
            "LLM_DEVICE": "cuda" if os.getenv("IMAGE_GPU", None) else "cpu",
            "TTS_LM_GENERATOR_TAG": "llm_transformers_generator",
        }
    )

if generator == "llamacpp":
    tts_grpc_image = tts_grpc_image.apt_install("clang")
    if os.getenv("IMAGE_GPU", None):
        tts_grpc_image = tts_grpc_image.run_commands(
            "find /usr/ -name 'libcuda.so.*'",
            "echo $LD_LIBRARY_PATH",
            f"LD_LIBRARY_PATH=/usr/local/cuda-12.5/compat:$LD_LIBRARY_PATH CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python --extra-index-url='https://abetlen.github.io/llama-cpp-python/whl/cu125'",
        )
    else:
        tts_grpc_image = tts_grpc_image.pip_install(
            f"achatbot[llama_cpp]=={achatbot_version}",
        )
    tts_grpc_image = tts_grpc_image.env(
        {
            # https://huggingface.co/mradermacher/SparkTTS-LLM-GGUF
            "LLM_MODEL_PATH": f"/root/.achatbot/models/mradermacher/SparkTTS-LLM-GGUF/SparkTTS-LLM.{os.getenv('QUANT','Q8_0')}.gguf",
            "TTS_LM_GENERATOR_TAG": "llm_llamacpp_generator",
            "N_GPU_LAYERS": "-1" if os.getenv("IMAGE_GPU", None) else "0",
            "FLASH_ATTN": "1" if os.getenv("IMAGE_GPU", None) else "",
            "QUANT": os.getenv("QUANT", "Q8_0"),
        }
    )


if generator == "vllm":
    tts_grpc_image = tts_grpc_image.pip_install(
        f"achatbot[vllm]=={achatbot_version}",
        "flashinfer-python==0.2.0.post2",
        extra_options="--find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer/",
        extra_index_url="https://flashinfer.ai/whl/cu125/torch2.5/",
    ).env(
        {
            "VLLM_USE_V1": os.getenv("VLLM_USE_V1", "1"),
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "TTS_LM_GENERATOR_TAG": "llm_vllm_generator",
            "LLM_TORCH_DTYPE": os.getenv("DTYPE", "bfloat16"),
        }
    )

if generator == "sglang":
    tts_grpc_image = tts_grpc_image.pip_install(
        f"achatbot[sglang]=={achatbot_version}",
        extra_options="--find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer/",
        extra_index_url="https://flashinfer.ai/whl/cu125/torch2.5/",
    ).env(
        {
            "TTS_LM_GENERATOR_TAG": "llm_sglang_generator",
            "LLM_TORCH_DTYPE": os.getenv("DTYPE", "bfloat16"),
        }
    )

if generator == "trtllm" or generator == "trtllm_runner":
    tts_grpc_image = tts_grpc_image.pip_install(
        f"achatbot[trtllm]=={achatbot_version}",
        "flashinfer-python==0.2.0.post2",
        pre=True,
        extra_index_url="https://pypi.nvidia.com",
    )
    if generator == "trtllm":
        tts_grpc_image = tts_grpc_image.env(
            {
                "TLLM_LLMAPI_BUILD_CACHE": "1",
                "TTS_LM_GENERATOR_TAG": "llm_trtllm_generator",
                "LLM_TORCH_DTYPE": os.getenv("DTYPE", "bfloat16"),
            }
        )
    if generator == "trtllm_runner":
        tts_grpc_image = tts_grpc_image.env(
            {
                "TTS_LM_GENERATOR_TAG": "llm_trtllm_runner_generator",
                "LLM_TORCH_DTYPE": os.getenv("DTYPE", "bfloat16"),
                "LLM_MODEL_NAME_OR_PATH": "/root/.achatbot/trt_models/tts-spark/trt_engines_bfloat16",
            }
        )

HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)

VLLM_CACHE_DIR = "/root/.cache/vllm"
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

TRT_MODEL_DIR = "/root/.achatbot/trt_models"
trt_model_vol = modal.Volume.from_name("triton_trtllm_models", create_if_missing=True)

TRT_MODEL_CACHE_DIR = "/tmp/.cache/tensorrt_llm/llmapi/"
trt_model_cache_vol = modal.Volume.from_name("triton_trtllm_cache_models", create_if_missing=True)


@app.function(
    gpu=os.getenv("IMAGE_GPU", None),
    cpu=2.0,
    image=tts_grpc_image,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_dir,
        VLLM_CACHE_DIR: vllm_cache_vol,
        TRT_MODEL_DIR: trt_model_vol,
        TRT_MODEL_CACHE_DIR: trt_model_cache_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=100,
)
async def run_generator():
    from achatbot.modules.speech.tts import TTSEnvInit
    from achatbot.common.interface import ITts
    from achatbot.common.types import SessionCtx
    from achatbot.common.session import Session
    from achatbot.common.logger import Logger
    from achatbot.common.utils.helper import get_device

    import numpy as np
    import librosa
    import soundfile
    import torch

    import uuid
    import subprocess

    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    print("cude is_available:", torch.cuda.is_available())
    device = get_device()

    Logger.init(
        os.getenv("LOG_LEVEL", "info").upper(),
        app_name=os.getenv("TTS_TAG"),
        is_file=False,
        is_console=True,
    )

    generator_tag = os.getenv("TTS_LM_GENERATOR_TAG", "")
    quant = os.getenv("QUANT", "")
    tts_engine: ITts = TTSEnvInit.initTTSEngine()

    texts = os.getenv("TTS_TEXT").split("|")
    for idx, text in enumerate(texts):
        if text == "":
            continue
        session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)
        session.ctx.state["tts_text"] = text
        session.ctx.state["temperature"] = 0.95
        print(session.ctx)
        iter = tts_engine.synthesize(session)
        res = bytearray()
        times = []
        start_time = perf_counter()
        i = 0
        async for chunk in iter:
            times.append(perf_counter() - start_time)
            res.extend(chunk)
            print(i, len(chunk))
            i += 1
            start_time = perf_counter()
        if len(res) == 0:
            print(f"no result")
            continue
        print(
            f"generate fisrt chunk time: {times[0]} s, {len(res)} waveform cost time: {sum(times)} s, avg cost time: {sum(times)/len(res)}"
        )

        stream_info = tts_engine.get_stream_info()
        print(f"stream_info:{stream_info}")

        file_name = f"test_{tts_engine.TAG}_{generator_tag}_{quant}_{device}_{idx}"
        os.makedirs(ASSETS_DIR, exist_ok=True)
        file_path = os.path.join(ASSETS_DIR, f"{file_name}.wav")
        data = np.frombuffer(res, dtype=stream_info["np_dtype"])

        # 去除静音部分
        # data, _ = librosa.effects.trim(data, top_db=60)
        # todo: need eval (SIM)

        soundfile.write(file_path, data, stream_info["rate"])
        info = soundfile.info(file_path, verbose=True)
        print(info)

        result = f"generate fisrt chunk time: {times[0]} s, tts cost time {sum(times)} s, wav duration {info.duration} s, RTF: {sum(times)/info.duration}"
        print(result)

        file_path = os.path.join(ASSETS_DIR, f"{file_name}.log")
        with open(file_path, "w") as f:
            f.write(result)


"""
# tts_generator_spark

# transformers with cpu
TTS_TAG=tts_generator_spark modal run src/tts/run_generator_tts.py
# transformers with gpu cuda
TTS_TAG=tts_generator_spark IMAGE_GPU=T4 modal run src/tts/run_generator_tts.py

# f16 don't support
# llamacpp with cpu, quant Q8_0
GENERATOR_ENGINE=llamacpp TTS_TAG=tts_generator_spark QUANT=Q8_0 ACHATBOT_VERSION=0.0.9.post3 modal run src/tts/run_generator_tts.py
# llamacpp with cpu, quant Q4_K_M
GENERATOR_ENGINE=llamacpp TTS_TAG=tts_generator_spark QUANT=Q4_K_M ACHATBOT_VERSION=0.0.9.post3 modal run src/tts/run_generator_tts.py
# llamacpp with cpu, quant Q2_K
GENERATOR_ENGINE=llamacpp TTS_TAG=tts_generator_spark QUANT=Q2_K ACHATBOT_VERSION=0.0.9.post3 modal run src/tts/run_generator_tts.py
# llamacpp with gpu cuda, quant Q8_0 flash attention
GENERATOR_ENGINE=llamacpp TTS_TAG=tts_generator_spark QUANT=Q8_0 ACHATBOT_VERSION=0.0.9.post3 IMAGE_GPU=L4 modal run src/tts/run_generator_tts.py
# llamacpp with gpu cuda, quant Q4_K_M flash attention
GENERATOR_ENGINE=llamacpp TTS_TAG=tts_generator_spark QUANT=Q4_K_M ACHATBOT_VERSION=0.0.9.post3 IMAGE_GPU=L4 modal run src/tts/run_generator_tts.py
# llamacpp with gpu cuda, quant Q2_K
GENERATOR_ENGINE=llamacpp TTS_TAG=tts_generator_spark QUANT=Q2_K ACHATBOT_VERSION=0.0.9.post3 IMAGE_GPU=T4 modal run src/tts/run_generator_tts.py

# vllm with gpu cuda | bf16 | Using Flash Attention backend | Using FlashInfer for top-p & top-k sampling
GENERATOR_ENGINE=vllm TTS_TAG=tts_generator_spark ACHATBOT_VERSION=0.0.9.post3 IMAGE_GPU=L4 modal run src/tts/run_generator_tts.py
GENERATOR_ENGINE=vllm TTS_TAG=tts_generator_spark ACHATBOT_VERSION=0.0.9.post3 IMAGE_GPU=L40S modal run src/tts/run_generator_tts.py

# tensorrt-llm with gpu cuda | bf16 | Using FlashInfer Attention backend (flashinfer.jit)
GENERATOR_ENGINE=trtllm TTS_TAG=tts_generator_spark ACHATBOT_VERSION=0.0.9.post3 IMAGE_GPU=L40S modal run src/tts/run_generator_tts.py
# tensorrt-llm runner with gpu cuda | bf16 | Using FlashInfer Attention backend (flashinfer.jit)
GENERATOR_ENGINE=trtllm_runner TTS_TAG=tts_generator_spark ACHATBOT_VERSION=0.0.9.post3 IMAGE_GPU=L40S modal run src/tts/run_generator_tts.py
"""


@app.local_entrypoint()
def main():
    run_generator.remote()
