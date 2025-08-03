import os
import time
import subprocess
import asyncio

import modal

ASR_TAG = os.getenv("ASR_TAG", "whisper_transformers_torch_compile_asr")

app = modal.App(f"torch-compile-run-whisper-achatbot")

image = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "cmake", "ninja-build")
    .pip_install("wheel")
    .pip_install(
        "accelerate",
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "torchvision==0.21.0",
        "torchcodec==0.2.0",
    )
    .pip_install("huggingface_hub[hf_transfer]", "transformers==4.52.4")
    .pip_install("librosa", "soundfile")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # faster model transfers
            "TORCH_CUDA_ARCH_LIST": "8.0 8.9 9.0+PTX",
            "ASR_TAG": ASR_TAG,
            "TORCHDYNAMO_VERBOSE": "1",
            # "TORCH_LOGS": "+dynamo",
            "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",  # https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html
        }
    )
)

MODEL_DIR = "/root/.achatbot/models"
model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)

image = image.pip_install(
    f"achatbot=={os.getenv('ACHATBOT_VERSION', '0.0.24')}",
    extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
)


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    retries=0,
    image=image,
    volumes={
        MODEL_DIR: model_vol,
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

    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("which nvcc", shell=True)
    subprocess.run("nvcc --version", shell=True)
    print("cuda:", torch.version.cuda)
    print("_GLIBCXX_USE_CXX11_ABI", torch._C._GLIBCXX_USE_CXX11_ABI)

    print("torch:", torch.__version__)
    subprocess.run("pip show transformers", shell=True)
    subprocess.run("pip show achatbot", shell=True)

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

    asr: IAsr = ASREnvInit.initASREngine(
        os.getenv("ASR_TAG", "whisper_transformers_torch_compile_asr"), **kwargs
    )
    audio_file = os.path.join(ASSETS_DIR, "asr_example_zh.wav")
    session = Session(**SessionCtx("test_client_id", 16000, 2).__dict__)
    asr.set_audio_data(audio_file)
    res = await asr.transcribe(session)
    print(res)


async def run_transcribe(**kwargs):
    import librosa
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = os.path.join(
        MODEL_DIR, kwargs.get("model_name_or_path", "openai/whisper-large-v3-turbo")
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    inference_time = 0.0
    model.generation_config.max_new_tokens = 128
    torch.set_float32_matmul_precision("high")

    audio_file = os.path.join(ASSETS_DIR, "asr_example_zh.wav")
    audio_np, _ = librosa.load(audio_file, sr=16000)

    # 1. Pre-process the audio inputs
    # input_features: torch.Size([1, 80, 3000]) torch.float32 for v2 v3 support 128
    input_features = processor(audio_np, sampling_rate=16000, return_tensors="pt").input_features
    print(input_features.shape, input_features.dtype)
    input_features = input_features.to(device, dtype=torch_dtype)
    # Create an attention mask
    attention_mask = torch.ones(input_features.shape[:2], dtype=torch.long, device=device)
    # 2. Auto-regressively generate text tokens
    start = time.time()
    pred_ids = model.generate(input_features, attention_mask=attention_mask)
    inference_time = time.time() - start
    # 3. Post-process tokens to text
    pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True)
    print(pred_text[0].lower())
    print(f"torch.compile Before {inference_time=}")

    # use torch compile
    model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
    model.generation_config.cache_implementation = "static"
    max_new_tokens = model.generation_config.max_new_tokens
    for i in range(3):
        start = time.time()
        with sdpa_kernel(SDPBackend.MATH):
            model.generate(
                input_features,
                attention_mask=attention_mask,
                min_new_tokens=max_new_tokens,
                max_new_tokens=max_new_tokens,
            )
        inference_time = time.time() - start
        print(f"warmup {i} step {inference_time=}")

    # 1. Pre-process the audio inputs
    # input_features: torch.Size([1, 80, 3000]) torch.float32 for v2 v3 support 128
    inputs = processor(
        audio_np,
        sampling_rate=16000,
        return_tensors="pt",
        padding="max_length",
        return_attention_mask=True,
    )
    inputs = inputs.to(device, dtype=torch_dtype)
    print(inputs)
    print(inputs.input_features.shape)

    # Create an attention mask
    attention_mask = torch.ones(inputs.input_features.shape[:2], dtype=torch.long, device=device)
    print(attention_mask.shape, inputs.attention_mask.shape)

    def gen(inputs: dict, **kwargs):
        with sdpa_kernel(SDPBackend.MATH):
            pred_ids = model.generate(**inputs, **kwargs)
            return pred_ids

    # 2. Auto-regressively generate text tokens
    start = time.time()

    # with sdpa_kernel(SDPBackend.MATH):
    #    pred_ids = model.generate(
    #        **inputs,
    #        # inputs.input_features,
    #        # attention_mask=attention_mask,
    #        task="transcribe",
    #    )

    pred_ids = await asyncio.to_thread(gen, inputs, task="transcribe")
    print(pred_ids)
    inference_time = time.time() - start
    # 3. Post-process tokens to text
    pred_text = processor.batch_decode(
        pred_ids, skip_special_tokens=True, decode_with_timestamps=True
    )
    print(pred_text[0].lower())
    print(f"torch.compile After {inference_time=}")


async def run_pipeline_transcribe(**kwargs):
    import librosa
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = os.path.join(
        MODEL_DIR, kwargs.get("model_name_or_path", "openai/whisper-large-v3-turbo")
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=1,  # batch size for inference - set based on your device
        torch_dtype=torch_dtype,
        device=device,
    )

    audio_file = os.path.join(ASSETS_DIR, "asr_example_zh.wav")
    audio_np, _ = librosa.load(audio_file, sr=16000)

    start = time.time()
    result = pipe(
        audio_np.copy(),
        chunk_length_s=30,
        batch_size=1,
        # generate_kwargs={"language": "zh", "task": "transcribe"},
        return_timestamps=True,
        # return_timestamps="word",
    )
    inference_time = time.time() - start
    print(result)
    print(f"torch.compile Before {inference_time=}")

    # 注意： torch.compile 目前与 Chunked 长格式算法或 Flash Attention 2 不兼容⚠️
    # Enable static cache and compile the forward pass
    model.generation_config.cache_implementation = "static"
    model.generation_config.max_new_tokens = 256
    model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    # 2 warmup steps
    for i in range(2):
        start = time.time()
        with sdpa_kernel(SDPBackend.MATH):
            result = pipe(
                audio_np.copy(),
                generate_kwargs={"min_new_tokens": 256, "max_new_tokens": 256},
                return_timestamps=True,
            )
        inference_time = time.time() - start
        print(f"warmup {i} step {inference_time=}")

    def pipe_gen(audio_np, **kwargs):
        with sdpa_kernel(SDPBackend.MATH):
            return pipe(audio_np, **kwargs)

    # fast run
    start = time.time()
    # with sdpa_kernel(SDPBackend.MATH):
    #    result = pipe(
    #        audio_np.copy(),
    #        chunk_length_s=30,
    #        batch_size=1,
    #        return_timestamps=True,
    #        # return_timestamps="word",
    #    )

    result = await asyncio.to_thread(
        pipe_gen,
        audio_np,
        generate_kwargs={
            "language": "zh",
            "task": "transcribe",
        },
        chunk_length_s=30,
        batch_size=1,
        return_timestamps=True,
        # return_timestamps="word",
    )
    inference_time = time.time() - start
    print(f"{inference_time=}")
    print(len(result["text"]), len(result["chunks"]))
    print(result["chunks"])


"""
https://huggingface.co/openai/whisper-large-v3
https://huggingface.co/openai/whisper-large-v3-turbo

# download model weights
modal run src/download_models.py --repo-ids "openai/whisper-large-v3"
modal run src/download_models.py --repo-ids "openai/whisper-large-v3-turbo"

ASR_TAG=whisper_transformers_torch_compile_asr modal run src/llm/achatbot_torch_compile_audio.py --task run_transcribe --model-name-or-path "openai/whisper-large-v3-turbo"
IMAGE_GPU=A100 ASR_TAG=whisper_transformers_torch_compile_asr modal run src/llm/achatbot_torch_compile_audio.py --task run_transcribe --model-name-or-path "openai/whisper-large-v3-turbo"

ASR_TAG=whisper_transformers_torch_compile_asr modal run src/llm/achatbot_torch_compile_audio.py --task run_transcribe --model-name-or-path "openai/whisper-large-v3"
IMAGE_GPU=A100 ASR_TAG=whisper_transformers_torch_compile_asr modal run src/llm/achatbot_torch_compile_audio.py --task run_transcribe --model-name-or-path "openai/whisper-large-v3"

ASR_TAG=whisper_transformers_torch_compile_asr modal run src/llm/achatbot_torch_compile_audio.py --task run_pipeline_transcribe --model-name-or-path "openai/whisper-large-v3-turbo"
IMAGE_GPU=A100 ASR_TAG=whisper_transformers_torch_compile_asr modal run src/llm/achatbot_torch_compile_audio.py --task run_transcribe --model-name-or-path "openai/whisper-large-v3-turbo"

ASR_TAG=whisper_transformers_torch_compile_asr modal run src/llm/achatbot_torch_compile_audio.py --task run_pipeline_transcribe --model-name-or-path "openai/whisper-large-v3"
IMAGE_GPU=A100 ASR_TAG=whisper_transformers_torch_compile_asr modal run src/llm/achatbot_torch_compile_audio.py --task run_transcribe --model-name-or-path "openai/whisper-large-v3"

# use large-v3-turbo model with torch compile
ASR_TAG=whisper_transformers_torch_compile_asr modal run src/llm/achatbot_torch_compile_audio.py --task transcribe --model-name-or-path "openai/whisper-large-v3-turbo"
ASR_TAG=whisper_transformers_pipeline_torch_compile_asr modal run src/llm/achatbot_torch_compile_audio.py --task transcribe --model-name-or-path "openai/whisper-large-v3-turbo"

# use large-v3 model with torch compile
ASR_TAG=whisper_transformers_torch_compile_asr modal run src/llm/achatbot_torch_compile_audio.py --task transcribe --model-name-or-path "openai/whisper-large-v3"
ASR_TAG=whisper_transformers_pipeline_torch_compile_asr modal run src/llm/achatbot_torch_compile_audio.py --task transcribe --model-name-or-path "openai/whisper-large-v3"

"""


@app.local_entrypoint()
def main(
    task: str = "transcribe",
    model_name_or_path: str = "openai/whisper-large-v3-turbo",
    # https://docs.pytorch.org/docs/stable/generated/torch.compile.html
    torch_compile_mode: str = "reduce-overhead",
):
    tasks = {
        "transcribe": transcribe,
        "run_transcribe": run_transcribe,
        "run_pipeline_transcribe": run_pipeline_transcribe,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    model_name_or_path = os.path.join(MODEL_DIR, model_name_or_path)
    run.remote(
        func=tasks[task],
        model_name_or_path=model_name_or_path,
        torch_compile_mode=torch_compile_mode,
    )
