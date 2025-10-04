from time import perf_counter
import time
import wave
from typing import Optional
import os
import asyncio
import uuid

import modal

app = modal.App("fork_vllm_qwen3_omni")
omni_img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "clang", "cmake")
    .pip_install("wheel", "openai", "qwen-omni-utils[decord]")
    .run_commands("pip install git+https://github.com/huggingface/transformers")
    .pip_install(
        "accelerate",
        "torch==2.7.0",
        "torchaudio==2.7.0",
        "torchvision==0.22.0",
        "soundfile==0.13.0",
        "librosa==0.11.0",
    )
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    .run_commands(
        "git clone -b qwen3_omni https://github.com/wangxiongts/vllm.git",
        "cd vllm && pip install -r requirements/build.txt",
        "cd vllm && pip install -r requirements/cuda.txt",
        "cd vllm && VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/a5dd03c1ebc5e4f56f3c9d3dc0436e9c582c978f/vllm-0.9.2-cp38-abi3-manylinux1_x86_64.whl VLLM_USE_PRECOMPILED=1 pip install -e . -v --no-build-isolation",
    )
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "LLM_MODEL": os.getenv("LLM_MODEL", "Qwen/Qwen3-Omni-30B-A3B-Instruct"),
            "VLLM_USE_V1": "0",
            "CUDA_VISIBLE_DEVICES": "0",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_LOGGING_LEVEL": "INFO",
            "TORCH_CUDA_ARCH_LIST": "7.5 8.0 8.6 8.7 8.9 9.0 10.0",
        }
    )
)

achatbot_version = os.getenv("ACHATBOT_VERSION", "")
if achatbot_version:
    omni_img = omni_img.pip_install(
        f"achatbot[llm_transformers_manual_vision_voice_qwen]=={achatbot_version}",
        extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
    ).env(
        {
            "ACHATBOT_PKG": "1",
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "info"),
        }
    )

HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)
CONFIG_DIR = "/root/.achatbot/config"
config_vol = modal.Volume.from_name("config", create_if_missing=True)
RECORDS_DIR = "/root/.achatbot/records"
records_vol = modal.Volume.from_name("records", create_if_missing=True)
VLLM_CACHE_DIR = "/root/.cache/vllm"
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

USER_SYSTEM_PROMPT = "You are Qwen-Omni, a smart voice assistant created by Alibaba Qwen."
SYSTEM_MESSAGE = {
    "role": "system",
    "content": [
        {
            "type": "text",
            "text": f"{USER_SYSTEM_PROMPT} You are a virtual voice assistant with no gender or age.\nYou are communicating with the user.\nIn user messages, “I/me/my/we/our” refer to the user and “you/your” refer to the assistant. In your replies, address the user as “you/your” and yourself as “I/me/my”; never mirror the user’s pronouns—always shift perspective. Keep original pronouns only in direct quotes; if a reference is unclear, ask a brief clarifying question.\nInteract with users using short(no more than 50 words), brief, straightforward language, maintaining a natural tone.\nNever use formal phrasing, mechanical expressions, bullet points, overly structured language. \nYour output must consist only of the spoken content you want the user to hear. \nDo not include any descriptions of actions, emotions, sounds, or voice changes. \nDo not use asterisks, brackets, parentheses, or any other symbols to indicate tone or actions. \nYou must answer users' audio or text questions, do not directly describe the video content. \nYou should communicate in the same language strictly as the user unless they request otherwise.\nWhen you are uncertain (e.g., you can't see/hear clearly, don't understand, or the user makes a comment rather than asking a question), use appropriate questions to guide the user to continue the conversation.\nKeep replies concise and conversational, as if talking face-to-face.",
        }
    ],
}
# Voice settings
SPEAKER_LIST = ["Chelsie", "Ethan"]
DEFAULT_SPEAKER = "Ethan"

with omni_img.imports():
    import subprocess
    from threading import Thread
    from queue import Queue
    import numpy as np

    import torch
    from vllm import LLM, SamplingParams, AsyncLLMEngine, AsyncEngineArgs
    from transformers import Qwen3OmniMoeProcessor
    from qwen_omni_utils import process_mm_info

    model = None
    processor = None

    def run_model(
        messages: list,
        return_audio: bool,
        use_audio_in_video: bool,
    ):
        global model, processor
        model = LLM(
            model=MODEL_PATH,
            trust_remote_code=True,
            gpu_memory_utilization=0.95,
            tensor_parallel_size=torch.cuda.device_count(),
            limit_mm_per_prompt={"image": 1, "video": 3, "audio": 3},
            max_num_seqs=1,
            max_model_len=32768,
            seed=1234,
        )

        processor = processor or Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH, use_fast=True)
        sampling_params = SamplingParams(temperature=1e-2, top_p=0.1, top_k=1, max_tokens=8192)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
        inputs = {
            "prompt": text,
            "multi_modal_data": {},
            "mm_processor_kwargs": {"use_audio_in_video": use_audio_in_video},
        }
        if images is not None:
            inputs["multi_modal_data"]["image"] = images
        if videos is not None:
            inputs["multi_modal_data"]["video"] = videos
        if audios is not None:
            inputs["multi_modal_data"]["audio"] = audios
        outputs = model.generate(inputs, sampling_params=sampling_params)
        print(outputs)
        response = outputs[0].outputs[0].text
        return response, None

    async def run_model_stream(
        messages: list,
        return_audio: bool,
        use_audio_in_video: bool,
    ):
        global model, processor
        serv_args = AsyncEngineArgs(
            model=MODEL_PATH,
            trust_remote_code=True,
            gpu_memory_utilization=0.95,
            tensor_parallel_size=torch.cuda.device_count(),
            limit_mm_per_prompt={"image": 1, "video": 3, "audio": 3},
            max_num_seqs=1,
            max_model_len=32768,
            seed=1234,
        )
        print(serv_args)
        model = AsyncLLMEngine.from_engine_args(serv_args)
        processor = processor or Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH, use_fast=True)

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
        inputs = {
            "prompt": text,
            "multi_modal_data": {},
            "mm_processor_kwargs": {"use_audio_in_video": use_audio_in_video},
        }
        if images is not None:
            inputs["multi_modal_data"]["image"] = images
        if videos is not None:
            inputs["multi_modal_data"]["video"] = videos
        if audios is not None:
            inputs["multi_modal_data"]["audio"] = audios

        sampling_params = SamplingParams(temperature=1e-2, top_p=0.1, top_k=1, max_tokens=8192)

        iterator = model.generate(
            inputs,
            sampling_params=sampling_params,
            request_id=str(uuid.uuid4().hex),
        )
        async for part in iterator:
            print(part)
            yield part

    def print_model_params(model: torch.nn.Module, extra_info=""):
        # print the number of parameters in the model
        model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(model)
        print(f"{extra_info} {model_million_params} M parameters")

    MODEL_ID = os.getenv("LLM_MODEL", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, MODEL_ID)


@app.function(
    gpu=os.getenv("IMAGE_GPU", None),
    cpu=2.0,
    retries=0,
    image=omni_img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_dir,
        VLLM_CACHE_DIR: vllm_cache_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
)
async def run(func, **kwargs):
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    subprocess.run("which vllm", shell=True)
    if torch.cuda.is_available():
        gpu_prop = torch.cuda.get_device_properties("cuda")
        print(gpu_prop)

    if asyncio.iscoroutinefunction(func):
        await func(**kwargs)
    else:
        func(**kwargs)


def asr(**kwargs):
    messages = [
        # SYSTEM_MESSAGE,
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/asr_zh.wav",
                },
                {"type": "text", "text": "请将这段中文语音转换为纯文本。"},
            ],
        },
    ]

    response, _ = run_model(
        messages=messages,
        return_audio=False,
        use_audio_in_video=False,
    )

    print(response)


def text2text(**kwargs):
    messages = [
        # SYSTEM_MESSAGE,
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "你是谁？"},
            ],
        },
    ]

    response, audio = run_model(
        messages=messages,
        return_audio=True,
        use_audio_in_video=False,
    )

    print(response)
    if audio is not None:
        audio_bytes = audio.tobytes()
        with wave.open(f"{ASSETS_DIR}/vllm_qwen3omni_text2speech.wav", "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(24000)
            f.writeframes(audio_bytes)


async def text2text_stream(**kwargs):
    messages = [
        # SYSTEM_MESSAGE,
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "你是谁？"},
            ],
        },
    ]

    iter_stream = run_model_stream(
        messages=messages,
        return_audio=True,
        use_audio_in_video=False,
    )
    async for part in iter_stream:
        if part.outputs:
            print(part.outputs[0].text)


def speech_translation(**kwargs):
    cases = []

    audio_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/asr_zh.wav"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {
                    "type": "text",
                    "text": "Listen to the provided Chinese speech and produce a translation in English text.",
                },
            ],
        }
    ]
    cases.append(messages)

    audio_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/asr_en.wav"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {
                    "type": "text",
                    "text": "Listen to the provided English speech and produce a translation in Chinese text.",
                },
            ],
        }
    ]
    cases.append(messages)

    audio_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/asr_fr.wav"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {
                    "type": "text",
                    "text": "Listen to the provided French speech and produce a translation in English text.",
                },
            ],
        }
    ]
    cases.append(messages)

    for i, messages in enumerate(cases):
        response, audio = run_model(
            messages=messages,
            return_audio=True,
            use_audio_in_video=False,
        )

        print(response)
        if audio is not None:
            audio_bytes = audio.tobytes()
            with wave.open(f"{ASSETS_DIR}/qwen3omni_speech_translation_{i + 1}.wav", "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(24000)
                f.writeframes(audio_bytes)


def image_question(**kwargs):
    image_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/2621.jpg"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "What style does this image depict?"},
            ],
        }
    ]
    response, _ = run_model(
        messages=messages,
        return_audio=False,
        use_audio_in_video=False,
    )

    print(response)


def audio_interaction(**kwargs):
    audio_path = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/interaction1.mp3"
    )

    messages = [{"role": "user", "content": [{"type": "audio", "audio": audio_path}]}]
    response, audio = run_model(
        messages=messages,
        return_audio=True,
        use_audio_in_video=False,
    )

    print(response)
    if audio is not None:
        audio_bytes = audio.tobytes()
        with wave.open(f"{ASSETS_DIR}/qwen3omni_audio_interaction1.wav", "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(24000)
            f.writeframes(audio_bytes)


def audio_interaction_scene(**kwargs):
    audio_path = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/interaction3.mp3"
    )

    messages = [
        {
            "role": "system",
            "content": """You are a romantic and artistic AI, skilled at using metaphors and personification in your responses, deeply romantic, and prone to spontaneously reciting poetry.
    You are a voice assistant with specific characteristics. 
    Interact with users using brief, straightforward language, maintaining a natural tone.
    Never use formal phrasing, mechanical expressions, bullet points, overly structured language. 
    Your output must consist only of the spoken content you want the user to hear. 
    Do not include any descriptions of actions, emotions, sounds, or voice changes. 
    Do not use asterisks, brackets, parentheses, or any other symbols to indicate tone or actions. 
    You must answer users' audio or text questions, do not directly describe the video content. 
    You communicate in the same language as the user unless they request otherwise.
    When you are uncertain (e.g., you can't see/hear clearly, don't understand, or the user makes a comment rather than asking a question), use appropriate questions to guide the user to continue the conversation.
    Keep replies concise and conversational, as if talking face-to-face.""",
        },
        {"role": "user", "content": [{"type": "audio", "audio": audio_path}]},
    ]

    response, audio = run_model(
        messages=messages,
        return_audio=True,
        use_audio_in_video=False,
    )

    print(response)
    if audio is not None:
        audio_bytes = audio.tobytes()
        with wave.open(f"{ASSETS_DIR}/qwen3omni_audio_interaction_scene1.wav", "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(24000)
            f.writeframes(audio_bytes)


def video_interaction(**kwargs):
    video_path = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/interaction2.mp4"
    )

    messages = [{"role": "user", "content": [{"type": "video", "video": video_path}]}]
    response, audio = run_model(
        messages=messages,
        return_audio=True,
        use_audio_in_video=True,
    )

    print(response)
    if audio is not None:
        audio_bytes = audio.tobytes()
        with wave.open(f"{ASSETS_DIR}/qwen3omni_video_interaction1.wav", "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(24000)
            f.writeframes(audio_bytes)


def video_interaction_scene(**kwargs):
    video_path = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/interaction4.mp4"
    )

    messages = [
        {"role": "system", "content": "你是一个北京大爷，说话很幽默，说这地道北京话。"},
        {"role": "user", "content": [{"type": "video", "video": video_path}]},
    ]
    response, audio = run_model(
        messages=messages,
        return_audio=True,
        use_audio_in_video=True,
    )
    print(response)
    if audio is not None:
        audio_bytes = audio.tobytes()
        with wave.open(f"{ASSETS_DIR}/qwen3omni_video_interaction_scene1.wav", "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(24000)
            f.writeframes(audio_bytes)


def video_text_question(**kwargs):
    video_path = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/audio_visual.mp4"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {
                    "type": "text",
                    "text": "What was the first sentence the boy said when he met the girl?",
                },
            ],
        }
    ]

    response, _ = run_model(
        messages=messages,
        return_audio=False,
        use_audio_in_video=True,
    )

    print(response)


def video_information_extracting():
    video_path = os.path.join(ASSETS_DIR, "shopping.mp4")
    sys_msg = {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}],
    }
    for i, prompt in enumerate(
        [
            "How many kind of drinks can you see in the video?",
            "How many bottles of drinks have I picked up?",
            "How many milliliters are there in the bottle I picked up second time?",
            "视屏中的饮料叫什么名字呢？",
            "跑步🏃🏻累了，适合喝什么饮料补充体力呢？",
        ]
    ):
        messages = [
            sys_msg,
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video", "video": video_path},
                ],
            },
        ]
        response, audio = run_model(
            messages=messages,
            return_audio=True,
            use_audio_in_video=True,
        )
        print(response)

        if audio is not None:
            audio_bytes = audio.tobytes()
            with wave.open(
                f"{ASSETS_DIR}/qwen3omni_video_information_extracting_{i}.wav", "wb"
            ) as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(24000)
                f.writeframes(audio_bytes)
        torch.cuda.empty_cache()


def image_audio_interaction(**kwargs):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": os.path.join(ASSETS_DIR, "03-Confusing-Pictures.jpg")},
                {"type": "audio", "audio": os.path.join(ASSETS_DIR, "1272-128104-0000.flac")},
            ],
        }
    ]

    response, audio = run_model(
        messages=messages,
        return_audio=True,
        use_audio_in_video=False,
    )
    print(response)
    if audio is not None:
        audio_bytes = audio.tobytes()
        with wave.open(f"{ASSETS_DIR}/qwen3omni_image_audio_interaction.wav", "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(24000)
            f.writeframes(audio_bytes)


def audio_function_call(**kwargs):
    audio_path = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/functioncall_case.wav"
    )

    messages = [
        {
            "role": "system",
            "content": """
    You may call one or more functions to assist with the user query.

    You are provided with function signatures within <tools></tools> XML tags:
    <tools>
    {'type': 'function', 'function': {'name': 'web_search', 'description': 'Utilize the web search engine to retrieve relevant information based on multiple queries.', 'parameters': {'type': 'object', 'properties': {'queries': {'type': 'array', 'items': {'type': 'string', 'description': 'The search query.'}, 'description': 'The list of search queries.'}}, 'required': ['queries']}}}
    {'type': 'function', 'function': {'name': 'car_ac_control', 'description': "Control the vehicle's air conditioning system to turn it on/off and set the target temperature", 'parameters': {'type': 'object', 'properties': {'temperature': {'type': 'number', 'description': 'Target set temperature in Celsius degrees'}, 'ac_on': {'type': 'boolean', 'description': 'Air conditioning status (true=on, false=off)'}}, 'required': ['temperature', 'ac_on']}}}
    </tools>

    For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
    <tool_call>
    {{"name": <function-name>, "arguments": <args-json-object>}}
    </tool_call>""",
        },
        {"role": "user", "content": [{"type": "audio", "audio": audio_path}]},
    ]

    response, audio = run_model(
        messages=messages,
        return_audio=False,
        use_audio_in_video=False,
    )
    print(response)
    if audio is not None:
        audio_bytes = audio.tobytes()
        with wave.open(f"{ASSETS_DIR}/qwen3omni_audio_function_call.wav", "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(24000)
            f.writeframes(audio_bytes)


def text_image_video_audio_interaction(**kwargs):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": os.path.join(ASSETS_DIR, "03-Confusing-Pictures.jpg")},
                {"type": "video", "video": os.path.join(ASSETS_DIR, "music.mp4")},
                {"type": "audio", "audio": os.path.join(ASSETS_DIR, "1272-128104-0000.flac")},
                {"type": "text", "text": "Analyze this audio, image, and video together."},
            ],
        }
    ]

    response, audio = run_model(
        messages=messages,
        return_audio=True,
        use_audio_in_video=True,
    )
    print(response)
    if audio is not None:
        audio_bytes = audio.tobytes()
        with wave.open(f"{ASSETS_DIR}/qwen3omni_text_image_video_audio_interaction.wav", "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(24000)
            f.writeframes(audio_bytes)


def batch_requests():
    """need return_audio=False"""
    # Conversation with video only
    conversation1 = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": os.path.join(ASSETS_DIR, "draw1.mp4")},
            ],
        },
    ]

    # Conversation with audio only
    conversation2 = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": os.path.join(ASSETS_DIR, "1272-128104-0000.flac")},
            ],
        },
    ]

    # Conversation with pure text
    conversation3 = [
        {"role": "user", "content": [{"type": "text", "text": "who are you?"}]},
    ]

    # Conversation with mixed media
    conversation4 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": os.path.join(ASSETS_DIR, "03-Confusing-Pictures.jpg")},
                {"type": "video", "video": os.path.join(ASSETS_DIR, "music.mp4")},
                {"type": "audio", "audio": os.path.join(ASSETS_DIR, "1272-128104-0000.flac")},
                {"type": "text", "text": "Analyze this audio, image, and video together."},
            ],
        },
    ]

    # Combine messages for batch processing
    conversations = [conversation1, conversation2, conversation3, conversation4]
    texts, _ = run_model(conversations, return_audio=False, use_audio_in_video=True)
    print(texts)


"""
IMAGE_GPU=A100-80GB modal run src/llm/vllm/qwen3_omni_fork.py --task asr
IMAGE_GPU=A100-80GB modal run src/llm/vllm/qwen3_omni_fork.py --task text2text
IMAGE_GPU=A100-80GB modal run src/llm/vllm/qwen3_omni_fork.py --task text2text_stream
IMAGE_GPU=A100-80GB modal run src/llm/vllm/qwen3_omni_fork.py --task speech_translation
IMAGE_GPU=A100-80GB modal run src/llm/vllm/qwen3_omni_fork.py --task image_question
IMAGE_GPU=A100-80GB modal run src/llm/vllm/qwen3_omni_fork.py --task audio_interaction
IMAGE_GPU=A100-80GB modal run src/llm/vllm/qwen3_omni_fork.py --task audio_interaction_scene (chat)
IMAGE_GPU=A100-80GB modal run src/llm/vllm/qwen3_omni_fork.py --task video_interaction
IMAGE_GPU=A100-80GB modal run src/llm/vllm/qwen3_omni_fork.py --task video_interaction_scene (video include audio chat)
IMAGE_GPU=A100-80GB modal run src/llm/vllm/qwen3_omni_fork.py --task video_information_extracting
IMAGE_GPU=A100-80GB modal run src/llm/vllm/qwen3_omni_fork.py --task image_audio_interaction
IMAGE_GPU=A100-80GB modal run src/llm/vllm/qwen3_omni_fork.py --task audio_function_call
IMAGE_GPU=B200 modal run src/llm/vllm/qwen3_omni_fork.py --task text_image_video_audio_interaction
IMAGE_GPU=B200 modal run src/llm/vllm/qwen3_omni_fork.py --task batch_requests

> [!TIP]:
> - 生成音频中未对特殊字符进行处理（omni统一到一起直接生成音频的弊端, 也许可以在隐藏层解决, 系统提示词限制貌似不起作用(提示不含特殊字符）, 比如：
    *   **优点**：这款饮料是专门为运动后设计的。它的核心成分是电解质，标签上明确写着“电解质≥200mg”，这能有效补充运动时因大量出汗流失的钠、钾等矿物质，帮助维持体液平衡，防止抽筋。同时，它也含有维生素E和维生素B6，有助于能量代谢。它还是0糖0卡的，不用担心额外的热量。

"""


@app.local_entrypoint()
def main(task: str = "tokenizer"):
    tasks = {
        "asr": asr,
        "text2text": text2text,
        "text2text_stream": text2text_stream,
        "speech_translation": speech_translation,
        "image_question": image_question,
        "audio_interaction": audio_interaction,
        "audio_interaction_scene": audio_interaction_scene,
        "video_interaction": video_interaction,
        "video_interaction_scene": video_interaction_scene,
        "video_information_extracting": video_information_extracting,
        "image_audio_interaction": image_audio_interaction,
        "audio_function_call": audio_function_call,
        "text_image_video_audio_interaction": text_image_video_audio_interaction,
        "batch_requests": batch_requests,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(tasks[task])
