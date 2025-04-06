import asyncio
import math
import os
from time import perf_counter
import modal


app = modal.App("tts-grpc")

achatbot_version = os.getenv("ACHATBOT_VERSION", "0.0.9.post4")

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
    if os.getenv("LLM_ATTN_IMPL") == "flash_attention_2":
        tts_grpc_image = tts_grpc_image.run_commands(
            "pip install flash-attn --no-build-isolation",
        )

    tts_grpc_image = tts_grpc_image.pip_install(
        f"achatbot[transformers]=={achatbot_version}",
        extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
    ).env(
        {
            "LLM_DEVICE": "cuda" if os.getenv("IMAGE_GPU", None) else "cpu",
            "TTS_LM_GENERATOR_TAG": "llm_transformers_generator",
            "LLM_ATTN_IMPL": os.getenv("LLM_ATTN_IMPL", "eager"),  # flash_attention_2,sdpa,eager
            "LLM_DEVICE_MAP": os.getenv("LLM_DEVICE_MAP", ""),  # auto
        }
    )

if generator == "llamacpp":
    tts_grpc_image = tts_grpc_image.apt_install("clang")
    if os.getenv("IMAGE_GPU", None):
        tts_grpc_image = tts_grpc_image.run_commands(
            "find /usr/ -name 'libcuda.so.*'",
            "echo $LD_LIBRARY_PATH",
            f"LD_LIBRARY_PATH=/usr/local/cuda-12.5/compat:$LD_LIBRARY_PATH CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python --extra-index-url='https://abetlen.github.io/llama-cpp-python/whl/cu125'",
            "pip install flash-attn --no-build-isolation",
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
        "accelerate",
        extra_options="--find-links https://flashinfer.ai/whl/cu125/torch2.5/flashinfer/",
        extra_index_url="https://flashinfer.ai/whl/cu125/torch2.5/",
    ).env(
        {
            "VLLM_USE_V1": os.getenv("VLLM_USE_V1", "1"),
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "TTS_LM_GENERATOR_TAG": "llm_vllm_generator",
            "LLM_TORCH_DTYPE": os.getenv("DTYPE", "bfloat16"),
            "LLM_GPU_MEMORY_UTILIZATION": os.getenv("LLM_GPU_MEMORY_UTILIZATION", "0.9"),
        }
    )

if generator == "sglang":
    tts_grpc_image = tts_grpc_image.pip_install(
        f"achatbot[sglang]=={achatbot_version}",
        extra_options="--find-links https://flashinfer.ai/whl/cu125/torch2.5/flashinfer/",
        extra_index_url="https://flashinfer.ai/whl/cu125/torch2.5/",
    ).env(
        {
            "TTS_LM_GENERATOR_TAG": "llm_sglang_generator",
            "LLM_TORCH_DTYPE": os.getenv("DTYPE", "bfloat16"),
            "LLM_GPU_MEMORY_UTILIZATION": os.getenv("LLM_GPU_MEMORY_UTILIZATION", "0.6"),
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
                # "TLLM_LLMAPI_BUILD_CACHE": "1",
                "TTS_LM_GENERATOR_TAG": "llm_trtllm_generator",
                "LLM_TORCH_DTYPE": os.getenv("DTYPE", "bfloat16"),
                "LLM_GPU_MEMORY_UTILIZATION": os.getenv("LLM_GPU_MEMORY_UTILIZATION", "0.7"),
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
    image=tts_grpc_image.pip_install("psutil").env(
        {
            "ACHATBOT_PKG": "1",
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "info"),
            "TORCH_CUDA_ARCH_LIST": "7.5 8.0 8.6 8.7 8.9 9.0",
            "TTS_TAG": os.getenv("TTS_TAG", "tts_generator_spark"),
            "WARMUP_TEXT": os.getenv(
                "WARMUP_TEXT",
                "hello,你好，我是机器人。|万物之始,大道至简,衍化至繁。",
            ),
            "TTS_TEXT": os.getenv(
                "TTS_TEXT",
                # "hello,你好，我是机器人。|万物之始,大道至简,衍化至繁。|君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。",
                "hello,你好，我是机器人。|万物之始,大道至简,衍化至繁。|君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。|PyTorch 将值组织成Tensor ， Tensor是具有丰富数据操作操作的通用 n 维数组。|Module 定义从输入值到输出值的转换，其在正向传递期间的行为由其forward成员函数指定。Module 可以包含Tensor作为参数。|例如，线性模块包含权重参数和偏差参数，其正向函数通过将输入与权重相乘并添加偏差来生成输出。|应用程序通过在自定义正向函数中将本机Module （*例如*线性、卷积等）和Function （例如relu、pool 等）拼接在一起来组成自己的Module 。|典型的训练迭代包含使用输入和标签生成损失的前向传递、用于计算参数梯度的后向传递以及使用梯度更新参数的优化器步骤。|更具体地说，在正向传递期间，PyTorch 会构建一个自动求导图来记录执行的操作。|然后，在反向传播中，它使用自动梯度图进行反向传播以生成梯度。最后，优化器应用梯度来更新参数。训练过程重复这三个步骤，直到模型收敛。",
            ),
            "CONCURRENCY_CN": os.getenv("CONCURRENCY_CN", "1"),
        }
    ),
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
    import platform

    import uuid
    import subprocess

    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    device = get_device()
    cpu = get_cup_info()
    gpu_arch = ""
    gpu_prop = None
    if torch.cuda.is_available():
        gpu_prop = torch.cuda.get_device_properties(device)
        print(gpu_prop)
        gpu_arch = f"{gpu_prop.major}.{gpu_prop.minor}"
        print(torch.cuda.memory_summary(device=device, abbreviated=True))
    else:
        print("CUDA is not available.")

    Logger.init(
        os.getenv("LOG_LEVEL", "info").upper(),
        app_name=os.getenv("TTS_TAG"),
        is_file=False,
        is_console=True,
    )

    generator_tag = os.getenv("TTS_LM_GENERATOR_TAG", "")
    concurrency_cn = int(os.getenv("CONCURRENCY_CN", "1"))
    quant = os.getenv("QUANT", "")
    tts_engine: ITts = TTSEnvInit.initTTSEngine()

    os.makedirs(ASSETS_DIR, exist_ok=True)
    result = (
        f"tts:{tts_engine.TAG}\n"
        + f"llm generator:{generator_tag}\n"
        + f"quant:{quant}\n"
        + f"cpu:{cpu}\ngpu:{gpu_prop}\n"
        + f"concurrency_cn:{concurrency_cn}\n"
    )
    file_name = f"test_{tts_engine.TAG}_{generator_tag}_{quant}_{device}_{gpu_arch}"
    stream_info = tts_engine.get_stream_info()
    print(f"stream_info:{stream_info}")

    async def generate(text, session=None, gen_global_token_ids=None):
        if session is None:
            session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)
            session.ctx.state["tts_text"] = text
            session.ctx.state["temperature"] = 0.95
        res = bytearray()
        times = []
        if text == "":
            return {"text": text, "waveform": res, "times": times, "gen_global_token_ids": []}
        session.ctx.state["gen_global_token_ids"] = gen_global_token_ids
        iter = tts_engine.synthesize(session)
        start_time = perf_counter()
        i = 0
        async for chunk in iter:
            times.append(perf_counter() - start_time)
            res.extend(chunk)
            print(i, len(chunk))
            i += 1
            start_time = perf_counter()
        if len(res) == 0:
            print("no waveform data")
            return {"text": text, "waveform": res, "times": times, "gen_global_token_ids": []}
        print(
            f"generate first chunk time: {times[0]} s, {len(res)} waveform cost time: {sum(times)} s"
        )
        if "gen_global_token_ids" in session.ctx.state:
            return {
                "text": text,
                "waveform": res,
                "times": times,
                "gen_global_token_ids": session.ctx.state["gen_global_token_ids"],
            }
        return {"text": text, "waveform": res, "times": times, "gen_global_token_ids": []}

    session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)
    warmup_texts = os.getenv("WARMUP_TEXT").split("|")
    gen_global_token_ids = None
    for text in warmup_texts:
        session.ctx.state["tts_text"] = text
        session.ctx.state["temperature"] = 0.95
        warmup_res = await generate(text, session, gen_global_token_ids=gen_global_token_ids)
        gen_global_token_ids = warmup_res.get("gen_global_token_ids")

    texts = os.getenv("TTS_TEXT").split("|")
    iter_cn = math.ceil(len(texts) / concurrency_cn)
    for idx in range(iter_cn):
        texts_chunk = texts[idx * concurrency_cn : (idx + 1) * concurrency_cn]
        tasks = [generate(text, gen_global_token_ids=gen_global_token_ids) for text in texts_chunk]
        start_time = perf_counter()
        gather_res = await asyncio.gather(*tasks)
        tts_cost_time = perf_counter() - start_time
        assert len(gather_res) == len(texts_chunk)

        waveform_bytes = bytearray()
        for res in gather_res:
            waveform_bytes.extend(res["waveform"])
        if len(waveform_bytes) == 0:
            print("no waveform data")
            continue
        file_path = os.path.join(ASSETS_DIR, f"{file_name}_{idx}.wav")
        data = np.frombuffer(waveform_bytes, dtype=stream_info["np_dtype"])

        # 去除静音部分
        # data, _ = librosa.effects.trim(data, top_db=60)
        # todo: need eval (WER|SIM)

        soundfile.write(file_path, data, stream_info["rate"])
        info = soundfile.info(file_path, verbose=True)
        print(info)

        text = "".join(texts_chunk)
        res = f"\n{text}\n"
        res += f"tts cost time {tts_cost_time} s, wav duration {info.duration} s, RTF: {tts_cost_time/info.duration}\n"
        res += "\n" + "--" * 10 + "\n"
        print(res)
        result += res

        if torch.cuda.is_available():
            total_memory = gpu_prop.total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            memory_usage_percent = (allocated_memory / total_memory) * 100
            print(
                f"PyTorch(exclude llm generator) GPU Memory Usage: {allocated_memory / (1024 ** 2):.2f} MB / {total_memory / (1024 ** 2):.2f} MB ({memory_usage_percent:.2f}%)"
            )

            print(torch.cuda.memory_summary(device=device, abbreviated=True))
            subprocess.run("nvidia-smi", shell=True)

    file_path = os.path.join(ASSETS_DIR, f"{file_name}.log")
    with open(file_path, "w") as f:
        f.write(result)

    return (result, file_name)


def get_cup_info():
    import psutil
    import platform

    # 获取CPU信息
    cpu_count = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq()
    cpu_times = psutil.cpu_times()

    cpu_info = {
        "cpu_arch": platform.machine(),
        "cpu_count_physical": cpu_count,
        "cpu_count_logical": cpu_count_logical,
        "cpu_freq_current": cpu_freq.current,
        "cpu_freq_min": cpu_freq.min,
        "cpu_freq_max": cpu_freq.max,
        "cpu_times_user": cpu_times.user,
        "cpu_times_system": cpu_times.system,
        "cpu_times_idle": cpu_times.idle,
    }
    return cpu_info


"""
# tts_generator_spark

# transformers with cpu
TTS_TAG=tts_generator_spark modal run src/tts/run_generator_tts.py
# transformers with gpu cuda eager
TTS_TAG=tts_generator_spark IMAGE_GPU=T4 LLM_ATTN_IMPL=eager modal run src/tts/run_generator_tts.py
# transformers with gpu cuda sdpa
TTS_TAG=tts_generator_spark IMAGE_GPU=T4 LLM_ATTN_IMPL=sdpa modal run src/tts/run_generator_tts.py
# transformers with gpu cuda flash_attention_2
TTS_TAG=tts_generator_spark IMAGE_GPU=L4 LLM_ATTN_IMPL=flash_attention_2 modal run src/tts/run_generator_tts.py
# transformers with auto device cuda flash_attention_2
TTS_TAG=tts_generator_spark IMAGE_GPU=L4 LLM_ATTN_IMPL=flash_attention_2 LLM_DEVICE_MAP=auto modal run src/tts/run_generator_tts.py

# f16 don't support
# llamacpp with cpu, quant Q8_0
GENERATOR_ENGINE=llamacpp TTS_TAG=tts_generator_spark QUANT=Q8_0 ACHATBOT_VERSION=0.0.9.post6 modal run src/tts/run_generator_tts.py
# llamacpp with cpu, quant Q4_K_M
GENERATOR_ENGINE=llamacpp TTS_TAG=tts_generator_spark QUANT=Q4_K_M ACHATBOT_VERSION=0.0.9.post6 modal run src/tts/run_generator_tts.py
# llamacpp with cpu, quant Q2_K
GENERATOR_ENGINE=llamacpp TTS_TAG=tts_generator_spark QUANT=Q2_K ACHATBOT_VERSION=0.0.9.post6 modal run src/tts/run_generator_tts.py
# llamacpp with gpu cuda, quant Q8_0 flash attention
GENERATOR_ENGINE=llamacpp TTS_TAG=tts_generator_spark QUANT=Q8_0 ACHATBOT_VERSION=0.0.9.post6 IMAGE_GPU=L4 modal run src/tts/run_generator_tts.py
# llamacpp with gpu cuda, quant Q4_K_M flash attention
GENERATOR_ENGINE=llamacpp TTS_TAG=tts_generator_spark QUANT=Q4_K_M ACHATBOT_VERSION=0.0.9.post6 IMAGE_GPU=L4 modal run src/tts/run_generator_tts.py
# llamacpp with gpu cuda, quant Q2_K
GENERATOR_ENGINE=llamacpp TTS_TAG=tts_generator_spark QUANT=Q2_K ACHATBOT_VERSION=0.0.9.post6 IMAGE_GPU=T4 modal run src/tts/run_generator_tts.py

# vllm with gpu cuda | bf16 | Using Flash Attention backend | Using FlashInfer for top-p & top-k sampling
GENERATOR_ENGINE=vllm TTS_TAG=tts_generator_spark ACHATBOT_VERSION=0.0.9.post6 IMAGE_GPU=L4 modal run src/tts/run_generator_tts.py
GENERATOR_ENGINE=vllm TTS_TAG=tts_generator_spark ACHATBOT_VERSION=0.0.9.post6 IMAGE_GPU=L40S modal run src/tts/run_generator_tts.py

# sglang with gpu cuda | bf16 | Using FlashInfer Attention backend (flashinfer.jit)
GENERATOR_ENGINE=sglang TTS_TAG=tts_generator_spark ACHATBOT_VERSION=0.0.9.post6 IMAGE_GPU=L40S modal run src/tts/run_generator_tts.py

# tensorrt-llm with gpu cuda | bf16
GENERATOR_ENGINE=trtllm TTS_TAG=tts_generator_spark ACHATBOT_VERSION=0.0.9.post6 IMAGE_GPU=L40S modal run src/tts/run_generator_tts.py
# tensorrt-llm runner with gpu cuda | bf16
GENERATOR_ENGINE=trtllm_runner TTS_TAG=tts_generator_spark ACHATBOT_VERSION=0.0.9.post6 IMAGE_GPU=L40S modal run src/tts/run_generator_tts.py

Tips: run trtllm_runner generator engine, if use diff gpu arch, need rebuild engine 

# CONCURRENCY_CN=4
CONCURRENCY_CN=4 GENERATOR_ENGINE=vllm TTS_TAG=tts_generator_spark ACHATBOT_VERSION=0.0.9.post6 IMAGE_GPU=L40S modal run src/tts/run_generator_tts.py
CONCURRENCY_CN=4 GENERATOR_ENGINE=sglang TTS_TAG=tts_generator_spark ACHATBOT_VERSION=0.0.9.post6 IMAGE_GPU=L40S modal run src/tts/run_generator_tts.py
CONCURRENCY_CN=4 GENERATOR_ENGINE=trtllm TTS_TAG=tts_generator_spark ACHATBOT_VERSION=0.0.9.post6 IMAGE_GPU=L40S modal run src/tts/run_generator_tts.py
CONCURRENCY_CN=4 GENERATOR_ENGINE=trtllm_runner TTS_TAG=tts_generator_spark ACHATBOT_VERSION=0.0.9.post6 IMAGE_GPU=L40S modal run src/tts/run_generator_tts.py
"""


@app.local_entrypoint()
def main():
    result, file_name = run_generator.remote()
    with open(f"{file_name}.log", "w") as f:
        f.write(result)
