import os
import subprocess
from threading import Thread
from time import perf_counter
import requests
import io
import uuid
import asyncio


import modal

APP_NAME = os.getenv("APP_NAME", "")

app = modal.App("gemma3")
img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "clang", "cmake", "ninja-build")
    .pip_install("wheel")
    .pip_install(
        "accelerate",
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "torchvision==0.21.0",
        "timm",
        "librosa",
    )
    # https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1
    # .pip_install("flash-attn==2.7.4.post1", extra_options="--no-build-isolation")
    .run_commands(
        "pip install git+https://github.com/huggingface/transformers.git@967045082faaaaf3d653bfe665080fd746b2bb60"
    )
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TQDM_DISABLE": "1",
            "LLM_MODEL": os.getenv("LLM_MODEL", "google/gemma-3n-E2B-it"),
        }
    )
)

if APP_NAME == "achatbot":
    img = img.pip_install(
        f"achatbot==0.0.21.post3",
        extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
    )

HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)


with img.imports():
    import random

    import torch
    from transformers import pipeline
    import numpy as np

    from PIL import Image
    import librosa
    from transformers import (
        AutoProcessor,
        Gemma3nForConditionalGeneration,
        DynamicCache,
    )
    from transformers.generation.streamers import TextIteratorStreamer

    torch.set_float32_matmul_precision("high")


@app.function(
    gpu=os.getenv("IMAGE_GPU", None),
    cpu=2.0,
    retries=1,
    image=img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_dir,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
)
async def run(func):
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    gpu_prop = torch.cuda.get_device_properties("cuda")
    print(gpu_prop)

    if asyncio.iscoroutinefunction(func):
        await func(gpu_prop)
    else:
        func(gpu_prop)


def print_model_params(model: torch.nn.Module, extra_info=""):
    # print the number of parameters in the model
    model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(model)
    print(f"{extra_info} {model_million_params} M parameters")


# gemama https://arxiv.org/abs/2403.08295
# gemama2 https://arxiv.org/abs/2408.00118
# gemma3 https://arxiv.org/abs/2503.19786
# https://huggingface.co/collections/google/gemma-3n-685065323f5984ef315c93f4
def dump_model(gpu_prop):
    for model_name in [
        "google/gemma-3n-E2B-it",
        "google/gemma-3n-E4B-it",
    ]:
        model_path = os.path.join(HF_MODEL_DIR, model_name)
        model = Gemma3nForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
        ).to("cuda")

        model = model.eval()
        print(f"{model.config=}")
        processor = AutoProcessor.from_pretrained(model_path)
        print(f"{processor=}")
        print_model_params(model, f"{model_name}")

        del model
        torch.cuda.empty_cache()


def generate(gpu_prop):
    # Load model
    model_name = os.getenv("LLM_MODEL")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
    model = Gemma3nForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
    )
    model = model.eval()
    print(f"{model.config=}")
    # print(f"{processor=}")
    # print(f"{model=}")
    print_model_params(model, f"{model_name}")

    text = "Describe this image in detail."
    text = "请用中文描述下图片内容"
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
                },
                {"type": "text", "text": text},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
                },
                {"type": "text", "text": text},
            ],
        },
    ]
    print(messages)

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)
    for key, value in inputs.items():
        print(f"{key}: {value.shape=}")
    input_ids = inputs["input_ids"]
    prompt = processor.decode(input_ids[0])
    print(f"{prompt=}")
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            top_k=1,
            top_p=0.9,
            repetition_penalty=1.1,
            max_new_tokens=256,
        )
        generated_ids = generated_ids[0][input_len:]

    print(f"{generated_ids.shape=}")
    generated_text = processor.decode(
        generated_ids,
        # skip_special_tokens=True,
    )
    print(generated_text)


def generate_audio(gpu_prop):
    # Load model
    model_name = os.getenv("LLM_MODEL")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
    model = Gemma3nForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
    )
    model = model.eval()
    # print(f"{model.config=}")
    # print(f"{processor=}")
    # print(f"{model=}")
    print_model_params(model, f"{model_name}")
    audio_path = os.path.join(ASSETS_DIR, "asr_example_zh.wav")
    audio_nparr, _ = librosa.load(audio_path, sr=16000, mono=True)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {
            "role": "user",
            "content": [
                # {"type": "audio", "audio": audio_path},
                {"type": "audio", "audio": audio_nparr},
                {
                    "type": "text",
                    "text": "Based on the attached audio, generate a comprehensive text transcription of the spoken content.",
                },
            ],
        },
    ]
    print(messages)

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)
    for key, value in inputs.items():
        print(f"{key}: {value.shape=}")
    input_ids = inputs["input_ids"]
    prompt = processor.decode(input_ids[0])
    print(f"{prompt=}")
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            top_k=1,
            top_p=0.9,
            repetition_penalty=1.1,
            max_new_tokens=256,
        )
        generated_ids = generated_ids[0][input_len:]

    print(f"{generated_ids.shape=}")
    generated_text = processor.decode(
        generated_ids,
        # skip_special_tokens=True,
    )
    print(generated_text)


def generate_stream(gpu_prop):
    # Load model
    MODEL_PATH = os.path.join(HF_MODEL_DIR, os.getenv("LLM_MODEL"))
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
    model = Gemma3nForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        # device_map="auto",
        # attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
    ).to("cuda")

    model = model.eval()
    # print(f"{model.config=}")
    # print(f"{processor=}")
    # print(f"{model=}")

    ## Construct prompt
    # image_file = os.path.join(ASSETS_DIR, "bee.jpg")
    ## Load and preprocess image
    # image = Image.open(image_file).convert("RGB")

    # url = "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"
    # image_bytes = requests.get(url).content
    # image = Image.open(io.BytesIO(image_bytes))

    image_1 = Image.new("RGB", (100, 100), color="white")
    image_2 = Image.new("RGB", (100, 100), color="blue")
    image_3 = Image.new("RGB", (100, 100), color="red")

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "你是一个中文语音智能助手，不要使用特殊字符回复，请使用中文回复。",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "图片是什么颜色"},
                {"type": "image", "image": image_1},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "这张图片是白色的。"}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "描述下图片"},
                {"type": "image", "image": image_2},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "图片中显示的是一个深蓝色的背景，上面有白色的字母“B”和“T”。",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "你叫什么名字"},
                {"type": "image", "image": image_3},
            ],
        },
    ]

    for i in range(3):
        inputs = processor.apply_chat_template(
            messages[: (i + 1) * 2],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)
        for key, value in inputs.items():
            print(f"{key}: {value.shape=}")
        input_ids = inputs["input_ids"]
        prompt = processor.decode(input_ids[0])
        print(f"{prompt=}")

        # kv cache
        # cache_position = torch.arange(input_ids.shape[1], dtype=torch.int64, device=model.device)
        # past_key_values = DynamicCache()

        streamer = TextIteratorStreamer(
            tokenizer=processor, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = dict(
            **inputs,
            # do_sample=False,
            # cache_position=cache_position,
            # past_key_values=past_key_values,
            # https://huggingface.co/docs/transformers/kv_cache
            # https://huggingface.co/docs/transformers/cache_explanation
            cache_implementation="dynamic",
            # cache_implementation="offloaded",
            do_sample=True,
            temperature=0.2,
            top_k=10,
            top_p=0.9,
            # num_beams=1,
            repetition_penalty=1.1,
            max_new_tokens=1024,
            use_cache=True,
            streamer=streamer,
        )
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        generated_text = ""
        start = perf_counter()
        times = []
        with torch.inference_mode():
            for new_text in streamer:
                times.append(perf_counter() - start)
                print(new_text, end="", flush=True)
                generated_text += new_text
                start = perf_counter()
        print(f"\n{i}. {generated_text=} TTFT: {times[0]:.2f}s total time: {sum(times):.2f}s")


def achatbot_gen_stream(gpu_prop):
    from achatbot.core.llm.transformers.manual_vision_speech_gemma3n import (
        TransformersManualVisionSpeechGemmaLM,
    )
    from achatbot.types.llm.transformers import TransformersLMArgs
    from achatbot.common.types import MODELS_DIR, SessionCtx
    from achatbot.common.session import Session
    from achatbot.common.logger import Logger

    Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)

    LLM_MODEL = os.getenv("LLM_MODEL")
    PATH = os.path.join(HF_MODEL_DIR, LLM_MODEL)
    model = TransformersManualVisionSpeechGemmaLM(
        **TransformersLMArgs(
            lm_model_name_or_path=PATH,
            init_chat_prompt="你是一个中文语音智能助手，不要使用特殊字符回复，请使用中文回复。",
            chat_history_size=10,
            lm_device="cuda",
            warmup_steps=1,
            warmup_prompt="描述下图片",
            lm_gen_temperature=0.6,
            lm_gen_repetition_penalty=1.1,
        ).__dict__
    )

    session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)

    url = "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"
    image_bytes = requests.get(url).content
    img = Image.open(io.BytesIO(image_bytes))

    audio_path = os.path.join(ASSETS_DIR, "asr_example_zh.wav")
    audio_nparr, _ = librosa.load(audio_path, sr=16000, mono=True)

    chat_texts = ["这张图片的内容是什么", "你叫什么名字", "讲个故事"]
    # chat_texts = ["这张图片的内容是什么"]
    for chat_text in chat_texts:
        session.ctx.state["prompt"] = [
            {"type": "image", "image": img},
            {"type": "audio", "audio": audio_nparr},
        ]
        for item in model.generate(session):
            result_text = item.get("text")
            print(result_text, flush=True, end="")

        session.ctx.state["prompt"] = [
            {"type": "image", "image": img},
            {"type": "text", "text": chat_text},
        ]
        for item in model.generate(session):
            result_text = item.get("text")
            print(result_text, flush=True, end="")

    chat_history = model.get_session_chat_history(session.ctx.client_id)
    print(f"\n{chat_history=}")
    img.close()


async def achatbot_asr(gpu_prop):
    from achatbot.modules.speech.asr.gemma3n_asr import Gemma3nAsr
    from achatbot.types.llm.transformers import TransformersLMArgs
    from achatbot.common.types import MODELS_DIR, SessionCtx
    from achatbot.common.session import Session
    from achatbot.common.logger import Logger

    Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)

    LLM_MODEL = os.getenv("LLM_MODEL")
    PATH = os.path.join(HF_MODEL_DIR, LLM_MODEL)
    asr = Gemma3nAsr(
        **TransformersLMArgs(
            lm_model_name_or_path=PATH,
            chat_history_size=0,
            lm_device="cuda",
            warmup_steps=0,
            warmup_prompt="描述下图片",
            lm_gen_temperature=0.6,
            lm_gen_repetition_penalty=1.1,
        ).__dict__
    )

    session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)

    audio_path = os.path.join(ASSETS_DIR, "asr_example_zh.wav")
    audio_nparr, _ = librosa.load(audio_path, sr=16000, mono=True)
    texts = [
        "Based on the attached audio, generate a comprehensive text transcription of the spoken content. ",
        "Based on the attached audio, generate a comprehensive text transcription of the spoken content. please use simple chinese",
    ]
    for text in texts:
        session.ctx.state["prompt"] = [
            {
                "type": "text",
                "text": text,
            },
            {"type": "audio", "audio": audio_nparr},
        ]

        async for text in asr.transcribe_stream(session):
            print(text, flush=True, end="")


"""
Gemma 3n:
Multimodal: 2b, 4b

IMAGE_GPU=T4 modal run src/llm/transformers/gemma3n.py --task dump_model

IMAGE_GPU=L4 modal run src/llm/transformers/gemma3n.py --task generate
IMAGE_GPU=L4 modal run src/llm/transformers/gemma3n.py --task generate_audio
IMAGE_GPU=L4 modal run src/llm/transformers/gemma3n.py --task generate_stream

APP_NAME=achatbot IMAGE_GPU=L4 modal run src/llm/transformers/gemma3n.py --task achatbot_gen_stream

APP_NAME=achatbot IMAGE_GPU=L4 modal run src/llm/transformers/gemma3n.py --task achatbot_asr
"""


@app.local_entrypoint()
def main(task: str = "dump_model"):
    print(task)
    tasks = {
        "dump_model": dump_model,
        "generate": generate,
        "generate_audio": generate_audio,
        "generate_stream": generate_stream,
        "achatbot_gen_stream": achatbot_gen_stream,
        "achatbot_asr": achatbot_asr,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(tasks[task])
