import io
import math
import requests
import os
import sys
import json
import time
import asyncio
import subprocess
from pathlib import Path
from threading import Thread


import modal


app = modal.App("step-audio")
IMAGE_GPU = os.getenv("IMAGE_GPU", None)
img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvcr.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs")
    .pip_install(
        "transformers==4.49.0",
        "torchaudio",
        "librosa",
        "onnxruntime",
        "s3tokenizer",
        "diffusers",
        "hyperpyyaml",
    )
    .run_commands(
        "git clone https://github.com/weedge/Step-Audio2.git -b main"
        " && cd /Step-Audio2"
        " && git checkout dac9ac36b157a80a64f3163452be26c6e41a76ac"
    )
    .pip_install("ffmpeg")
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "ACHATBOT_PKG": "1",
            "LLM_MODEL": os.getenv("LLM_MODEL", "stepfun-ai/Step-Audio-2-mini"),
        }
    )
)

# img = img.pip_install(
#    f"achatbot==0.0.25.dev122",
#    extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://test.pypi.org/simple/"),
# )


HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_vol = modal.Volume.from_name("assets", create_if_missing=True)
CONFIG_DIR = "/root/.achatbot/config"
config_vol = modal.Volume.from_name("config", create_if_missing=True)
RECORDS_DIR = "/root/.achatbot/records"
records_vol = modal.Volume.from_name("records", create_if_missing=True)

TORCH_CACHE_DIR = "/root/.cache/torch"
torch_cache_vol = modal.Volume.from_name("torch_cache", create_if_missing=True)


with img.imports():
    from queue import Queue

    import wave
    import torch
    from transformers import GenerationConfig
    from transformers.generation.streamers import BaseStreamer

    sys.path.insert(1, "/Step-Audio2")

    from stepaudio2 import StepAudio2, StepAudio2Base
    from token2wav import Token2wav
    from utils import compute_token_num, load_audio, log_mel_spectrogram, padding_mels

    MODEL_ID = os.getenv("LLM_MODEL", "stepfun-ai/Step-Audio-2-mini")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, MODEL_ID)
    os.makedirs(f"{ASSETS_DIR}/StepAudio2", exist_ok=True)

    CHUNK_SIZE = 25

    # torch.set_float32_matmul_precision("high")


def print_model_params(model: torch.nn.Module, extra_info="", f=None):
    # print the number of parameters in the model
    model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(model, file=f)
    print(f"{extra_info} {model_million_params} M parameters", file=f)


@app.function(
    gpu=IMAGE_GPU,
    cpu=2.0,
    retries=0,
    image=img,
    secrets=[modal.Secret.from_name("achatbot")],
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
)
async def run(func, **kwargs):
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    gpu_prop = None
    if torch.cuda.is_available():
        gpu_prop = torch.cuda.get_device_properties("cuda")
        print(gpu_prop)

    if asyncio.iscoroutinefunction(func):
        await func(gpu_prop, **kwargs)
    else:
        func(gpu_prop, **kwargs)


def dump_model(gpu_prop, **kwargs):
    if "Base" in MODEL_ID:
        model = StepAudio2Base(MODEL_PATH)  # mini-base
    else:
        model = StepAudio2(MODEL_PATH)  # mini
    # print_model_params(model.llm, MODEL_ID)  # AudioEncoder + Adpater + LLM decoder

    token2wav = Token2wav(f"{MODEL_PATH}/token2wav")

    print_model_params(
        token2wav.flow, f"{MODEL_ID}/token2wav.flow"
    )  # Flow-Matching audio_tokens->mels
    print_model_params(
        token2wav.flow.encoder, f"{MODEL_ID}/token2wav.flow.encoder"
    )  # Flow-Matching encoder
    print_model_params(
        token2wav.flow.decoder, f"{MODEL_ID}/token2wav.flow.decoder"
    )  # Flow-Matching decoder

    print_model_params(token2wav.hift, f"{MODEL_ID}/token2wav.hift")  # Vocoder mels->waveform

    print_model_params(
        token2wav.audio_tokenizer, f"{MODEL_ID}/token2wav.audio_tokenizer"
    )  # for ref audio quantization (FSQ)
    print(
        token2wav.spk_model, f"{MODEL_ID}/token2wav.spk_model"
    )  # for ref audio speaker embedding(fbank feat)

    # print(f"{model.llm_tokenizer=}")  # text tokenizer with instruct specail token
    print(f"{model.llm.config=}")

    print(model.llm_tokenizer.decode(49434))
    print(model.llm_tokenizer.decode(239))
    print(model.llm_tokenizer.decode([49434, 239]))


def tokenize(gpu_prop, **kwargs):
    if "Base" in MODEL_ID:
        model = StepAudio2Base(MODEL_PATH)  # mini-base
        messages = [
            "请记录下你所听到的语音内容。",
            {
                "type": "audio",
                "audio": "/Step-Audio2/assets/give_me_a_brief_introduction_to_the_great_wall.wav",
            },
        ]
    else:
        model = StepAudio2(MODEL_PATH)  # mini
        messages = [
            {"role": "system", "content": "请记录下你所听到的语音内容。"},
            {
                "role": "human",
                "content": [
                    {
                        "type": "audio",
                        "audio": "/Step-Audio2/assets/give_me_a_brief_introduction_to_the_great_wall.wav",
                    }
                ],
            },
            {"role": "assistant", "content": None},
        ]

    print(model.llm_tokenizer, model.eos_token_id)

    res, mels = model.apply_chat_template(messages)
    print(res)
    print(mels)

    # Tokenize prompts
    prompt_ids = []
    for msg in res:
        if isinstance(msg, str):
            prompt_ids.append(
                model.llm_tokenizer(text=msg, return_tensors="pt", padding=True)["input_ids"]
            )
        elif isinstance(msg, list):
            prompt_ids.append(torch.tensor([msg], dtype=torch.int32))
        else:
            raise ValueError(f"Unsupported content type: {type(msg)}")
    prompt_ids = torch.cat(prompt_ids, dim=-1).cuda()
    attention_mask = torch.ones_like(prompt_ids)
    print(prompt_ids)
    print(attention_mask)

    # mels = None if len(mels) == 0 else torch.stack(mels).cuda()
    # mel_lengths = None if mels is None else torch.tensor([mel.shape[1] - 2 for mel in mels], dtype=torch.int32, device='cuda')
    mels, mel_lengths = padding_mels(mels)
    print(mels, mel_lengths)


# ASR
def asr_test(model, token2wav=None):
    messages = [
        "请记录下你所听到的语音内容。",
        {
            "type": "audio",
            "audio": "/Step-Audio2/assets/give_me_a_brief_introduction_to_the_great_wall.wav",
        },
    ]
    eos_token_id = model.llm_tokenizer.convert_tokens_to_ids("<|endoftext|>")
    tokens, text, _ = model(
        messages,
        max_new_tokens=256,
        temperature=0.1,
        do_sample=True,
        eos_token_id=[model.eos_token_id, eos_token_id],
    )
    print(text)


# S2TT（support: en,zh,ja）
def s2tt_test(model, token2wav=None):
    messages = [
        "请仔细聆听这段语音，然后将其内容翻译成中文",
        # "Please listen carefully to this audio and then translate its content into Chinese.",
        {
            "type": "audio",
            "audio": "/Step-Audio2/assets/give_me_a_brief_introduction_to_the_great_wall.wav",
        },
    ]
    eos_token_id = model.llm_tokenizer.convert_tokens_to_ids("<|endoftext|>")
    tokens, text, _ = model(
        messages,
        max_new_tokens=256,
        temperature=0.1,
        do_sample=True,
        eos_token_id=[model.eos_token_id, eos_token_id],
    )
    print(text)


# audio caption
def audio_caption_test(model, token2wav=None):
    messages = [
        "Please briefly explain the important events involved in this audio clip.",
        {
            "type": "audio",
            "audio": "/Step-Audio2/assets/music_playing_followed_by_a_woman_speaking.wav",
        },
    ]
    eos_token_id = model.llm_tokenizer.convert_tokens_to_ids("<|endoftext|>")
    tokens, text, _ = model(
        messages,
        max_new_tokens=256,
        temperature=0.1,
        do_sample=True,
        eos_token_id=[model.eos_token_id, eos_token_id],
    )
    print(text)


# TTS（support: en,zh,ja)
def tts_test(model, token2wav):
    messages = [
        "以自然的语速读出下面的文字。\n",
        # "Read this paragraph at a natural pace.\n",
        "你好呀，我是你的AI助手，很高兴认识你！<tts_start>",
    ]
    tokens, text, audio = model(messages, max_tokens=2048, temperature=0.7, do_sample=True)
    print(text)
    # print(tokens)
    audio = [x for x in audio if x < 6561]  # remove audio padding
    audio = token2wav(audio, prompt_wav="/Step-Audio2/assets/default_male.wav")
    with open(f"{ASSETS_DIR}/StepAudio2/output-tts.wav", "wb") as f:
        f.write(audio)


# T2ST（support: en,zh,ja)
def t2st_test(model, token2wav):
    messages = [
        "将下面的文本翻译成英文，并用语音播报。\n",
        # "Translate the following text into English and broadcast it with voice.\n",
        "你好呀，我是你的AI助手，很高兴认识你！<tts_start>",
    ]
    tokens, text, audio = model(messages, max_tokens=2048, temperature=0.7, do_sample=True)
    print(text)
    # print(tokens)
    audio = [x for x in audio if x < 6561]  # remove audio padding
    audio = token2wav(audio, prompt_wav="/Step-Audio2/assets/default_male.wav")
    with open(f"{ASSETS_DIR}/StepAudio2/output-t2st.wav", "wb") as f:
        f.write(audio)


# S2ST（support: en,zh）
def s2st_test(model, token2wav):
    messages = [
        "请仔细聆听这段语音，然后将其内容翻译成中文并用语音播报。",
        # "Please listen carefully to this audio and then translate its content into Chinese speech.",
        {
            "type": "audio",
            "audio": "/Step-Audio2/assets/give_me_a_brief_introduction_to_the_great_wall.wav",
        },
        "<tts_start>",
    ]
    tokens, text, audio = model(messages, max_tokens=2048, temperature=0.7, do_sample=True)
    print(text)
    # print(tokens)
    audio = [x for x in audio if x < 6561]  # remove audio padding
    audio = token2wav(audio, prompt_wav="/Step-Audio2/assets/default_female.wav")
    with open(f"{ASSETS_DIR}/StepAudio2/output-s2st.wav", "wb") as f:
        f.write(audio)


# multi turn aqta
def multi_turn_aqta_test(model, token2wav=None):
    history = []
    for round_idx, inp_audio in enumerate(
        [
            "/Step-Audio2/assets/multi-turn-round1-听说荡口古镇从下个月开始取消门票了，你知道这事吗。.wav",
            "/Step-Audio2/assets/multi-turn-round2-新闻说九月十九号就免费开放了。好像整个古镇都升级改造了，现在变成开放式街区了。.wav",
        ]
    ):
        print("round: ", round_idx)
        history.append({"type": "audio", "audio": inp_audio})
        tokens, text, _ = model(history, max_new_tokens=256, temperature=0.5, do_sample=True)
        print(text)
        history.append(text)


# multi turn aqaa
def multi_turn_aqaa_test(model, token2wav):
    history = []
    for round_idx, inp_audio in enumerate(
        [
            "/Step-Audio2/assets/multi-turn-round1-听说荡口古镇从下个月开始取消门票了，你知道这事吗。.wav",
            "/Step-Audio2/assets/multi-turn-round2-新闻说九月十九号就免费开放了。好像整个古镇都升级改造了，现在变成开放式街区了。.wav",
        ]
    ):
        print("round: ", round_idx)
        history.append(
            {"type": "audio", "audio": inp_audio},
        )
        history.append("<tts_start>")
        tokens, text, audio = model(history, max_new_tokens=2048, temperature=0.7, do_sample=True)
        print(text)
        audio = [x for x in audio if x < 6561]  # remove audio padding
        audio = token2wav(audio, prompt_wav="/Step-Audio2/assets/default_female.wav")
        with open(f"{ASSETS_DIR}/StepAudio2/output-round-{round_idx}.wav", "wb") as f:
            f.write(audio)
        history.append({"type": "token", "token": tokens})


def test_base(gpu_prop, **kwargs):
    model = StepAudio2Base(MODEL_PATH)
    token2wav = Token2wav(f"{MODEL_PATH}/token2wav")

    test_func = kwargs.get("test_func", "asr_test")
    globals()[test_func](model, token2wav)


# -------------------------------------------------------------------------------------------------
# special Instruct


# ASR
def instruct_asr_test(model, token2wav):
    messages = [
        {"role": "system", "content": "请记录下你所听到的语音内容。"},
        # {"role": "system", "content": "Please record the audio content you hear."},
        {
            "role": "human",
            "content": [
                {
                    "type": "audio",
                    "audio": "/Step-Audio2/assets/give_me_a_brief_introduction_to_the_great_wall.wav",
                }
            ],
        },
        {"role": "assistant", "content": None},
    ]
    tokens, text, _ = model(messages, max_new_tokens=256)
    print(text)


# audio caption
def instruct_audio_caption_test(model, token2wav):
    messages = [
        {
            "role": "system",
            "content": "Please briefly explain the important events involved in this audio clip.",
        },
        {
            "role": "human",
            "content": [
                {
                    "type": "audio",
                    "audio": "/Step-Audio2/assets/music_playing_followed_by_a_woman_speaking.wav",
                }
            ],
        },
        {"role": "assistant", "content": None},
    ]
    tokens, text, _ = model(messages, max_new_tokens=256, temperature=0.1, do_sample=True)
    print(text)


# S2TT（support: en,zh,ja）
def instruct_s2tt_test(model, token2wav):
    messages = [
        {"role": "system", "content": "请仔细聆听这段语音，然后将其内容翻译成中文。"},
        # {"role": "system", "content":"Please listen carefully to this audio and then translate its content into Chinese."},
        {
            "role": "human",
            "content": [
                {
                    "type": "audio",
                    "audio": "/Step-Audio2/assets/give_me_a_brief_introduction_to_the_great_wall.wav",
                }
            ],
        },
        {"role": "assistant", "content": None},
    ]
    tokens, text, _ = model(messages, max_new_tokens=256, temperature=0.1, do_sample=True)
    print(text)


# S2ST（support: en,zh）
def instruct_s2st_test(model, token2wav):
    messages = [
        {"role": "system", "content": "请仔细聆听这段语音，然后将其内容翻译成中文并用语音播报。"},
        # {"role": "system", "content":"Please listen carefully to this audio and then translate its content into Chinese speech."},
        {
            "role": "human",
            "content": [
                {
                    "type": "audio",
                    "audio": "/Step-Audio2/assets/give_me_a_brief_introduction_to_the_great_wall.wav",
                }
            ],
        },
        {
            "role": "assistant",
            "content": "<tts_start>",
            "eot": False,
        },  # Insert <tts_start> for speech response
    ]
    tokens, text, audio = model(messages, max_tokens=2048, temperature=0.7, do_sample=True)
    print(text)
    # print(tokens)
    audio = [x for x in audio if x < 6561]  # remove audio padding
    audio = token2wav(audio, prompt_wav="/Step-Audio2/assets/default_female.wav")
    with open(f"{ASSETS_DIR}/StepAudio2/output-s2st.wav", "wb") as f:
        f.write(audio)


# multi turn tqta
def instruct_multi_turn_tqta_test(model, token2wav):
    history = [{"role": "system", "content": "You are a helpful assistant."}]
    for round_idx, input_text in enumerate(
        [
            "听说荡口古镇从下个月开始取消门票了，你知道这事吗。",
            "新闻说九月十九号就免费开放了。好像整个古镇都升级改造了，现在变成开放式街区了。",
        ]
    ):
        print("round: ", round_idx)
        history.append({"role": "human", "content": [{"type": "text", "text": input_text}]})
        history.append({"role": "assistant", "content": None})
        tokens, text, _ = model(history, max_new_tokens=256, temperature=0.5, do_sample=True)
        print(text)
        history.pop(-1)
        history.append({"role": "assistant", "content": text})


# multi turn tqaa
def instruct_multi_turn_tqaa_test(model, token2wav):
    history = [{"role": "system", "content": "You are a helpful assistant."}]
    for round_idx, input_text in enumerate(
        [
            "听说荡口古镇从下个月开始取消门票了，你知道这事吗。",
            "新闻说九月十九号就免费开放了。好像整个古镇都升级改造了，现在变成开放式街区了。",
        ]
    ):
        print("round: ", round_idx)
        history.append({"role": "human", "content": [{"type": "text", "text": input_text}]})
        history.append(
            {
                "role": "assistant",
                "content": "<tts_start>",
                "eot": False,
            },  # Insert <tts_start> for speech response
        )
        tokens, text, audio = model(history, max_new_tokens=256, temperature=0.5, do_sample=True)
        print(tokens, model.llm_tokenizer.decode(tokens))
        print(text)
        audio = [x for x in audio if x < 6561]  # remove audio padding
        audio = token2wav(audio, prompt_wav="/Step-Audio2/assets/default_female.wav")
        with open(f"{ASSETS_DIR}/StepAudio2/output-round-tqaa-{round_idx}.wav", "wb") as f:
            f.write(audio)
        history.pop(-1)
        history.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "<tts_start>"},
                    {"type": "token", "token": tokens},
                ],
            }
        )


# multi turn aqta
def instruct_multi_turn_aqta_test(model, token2wav):
    history = [{"role": "system", "content": "You are a helpful assistant."}]
    for round_idx, inp_audio in enumerate(
        [
            "/Step-Audio2/assets/multi-turn-round1-听说荡口古镇从下个月开始取消门票了，你知道这事吗。.wav",
            "/Step-Audio2/assets/multi-turn-round2-新闻说九月十九号就免费开放了。好像整个古镇都升级改造了，现在变成开放式街区了。.wav",
        ]
    ):
        print("round: ", round_idx)
        history.append({"role": "human", "content": [{"type": "audio", "audio": inp_audio}]})
        history.append({"role": "assistant", "content": None})
        tokens, text, _ = model(history, max_new_tokens=256, temperature=0.5, do_sample=True)
        print(text)
        history.pop(-1)
        history.append({"role": "assistant", "content": text})


# multi turn aqaa
def instruct_multi_turn_aqaa_test(model, token2wav):
    history = [{"role": "system", "content": "You are a helpful assistant."}]
    for round_idx, inp_audio in enumerate(
        [
            "/Step-Audio2/assets/multi-turn-round1-听说荡口古镇从下个月开始取消门票了，你知道这事吗。.wav",
            "/Step-Audio2/assets/multi-turn-round2-新闻说九月十九号就免费开放了。好像整个古镇都升级改造了，现在变成开放式街区了。.wav",
        ]
    ):
        print("round: ", round_idx)
        history.append({"role": "human", "content": [{"type": "audio", "audio": inp_audio}]})
        history.append(
            {
                "role": "assistant",
                "content": "<tts_start>",
                "eot": False,
            },  # Insert <tts_start> for speech response
        )
        tokens, text, audio = model(history, max_new_tokens=2048, temperature=0.7, do_sample=True)
        print(tokens, model.llm_tokenizer.decode(tokens))
        print(text)
        audio = [x for x in audio if x < 6561]  # remove audio padding
        audio = token2wav(audio, prompt_wav="/Step-Audio2/assets/default_female.wav")
        with open(f"{ASSETS_DIR}/StepAudio2/output-round-aqaa-{round_idx}.wav", "wb") as f:
            f.write(audio)
        history.pop(-1)
        history.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "<tts_start>"},
                    {"type": "token", "token": tokens},
                ],
            }
        )


# Tool call & Web search
def instruct_tool_call_test(model, token2wav):
    history = [
        {
            "role": "system",
            "content": "你的名字叫做小跃，是由阶跃星辰公司训练出来的语音大模型。\n你具备调用工具解决问题的能力，你需要根据用户的需求和上下文情景，自主选择是否调用系统提供的工具来协助用户。\n你情感细腻，观察能力强，擅长分析用户的内容，并作出善解人意的回复，说话的过程中时刻注意用户的感受，富有同理心，提供多样的情绪价值。\n今天是2025年8月28日，星期四\n请用默认女声与用户交流",
        },
        {
            "role": "tool_json_schemas",
            "content": '[{"type": "function", "function": {"name": "search", "description": "搜索工具", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "搜索关键词"}}, "required": ["query"], "additionalProperties": false}}}]',
        },
        {
            "role": "human",
            "content": [
                {
                    "type": "audio",
                    "audio": "/Step-Audio2/assets/帮我查一下今天上证指数的开盘价是多少.wav",
                }
            ],
        },
        {
            "role": "assistant",
            "content": "<tts_start>",
            "eot": False,
        },  # Insert <tts_start> for speech response
    ]
    tokens, text, audio = model(
        history,
        max_new_tokens=4096,
        repetition_penalty=1.05,
        top_p=0.9,
        temperature=0.7,
        do_sample=True,
    )
    print(text)
    audio = [x for x in audio if x < 6561]  # remove audio padding
    audio = token2wav(audio, prompt_wav="/Step-Audio2/assets/default_female.wav")
    with open(f"{ASSETS_DIR}/StepAudio2/output-tool-call-1.wav", "wb") as f:
        f.write(audio)
    history.pop(-1)
    with open("/Step-Audio2/assets/search_result.txt") as f:
        search_result = f.read().strip()
        print(f"search result: {search_result}")
    history += [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "<tts_start>"},
                {"type": "token", "token": tokens},
            ],
        },
        {
            "role": "input",
            "content": [
                {"type": "text", "text": search_result},
                {
                    "type": "text",
                    "text": "\n\n\n请用口语化形式总结检索结果，简短地回答用户的问题。",
                },
            ],
        },
        {
            "role": "assistant",
            "content": "<tts_start>",
            "eot": False,
        },  # Insert <tts_start> for speech response
    ]
    tokens, text, audio = model(
        history,
        max_new_tokens=4096,
        repetition_penalty=1.05,
        top_p=0.9,
        temperature=0.7,
        do_sample=True,
    )
    print(text)
    audio = [x for x in audio if x < 6561]  # remove audio padding
    audio = token2wav(audio, prompt_wav="/Step-Audio2/assets/default_female.wav")
    with open(f"{ASSETS_DIR}/StepAudio2/output-tool-call-2.wav", "wb") as f:
        f.write(audio)


# Paralingustic information understanding
def instruct_paralinguistic_test(model, token2wav):
    messages = [
        {"role": "system", "content": "请用语音与我交流。"},
        {
            "role": "human",
            "content": [
                {
                    "type": "audio",
                    "audio": "/Step-Audio2/assets/paralinguistic_information_understanding.wav",
                }
            ],
        },
        {
            "role": "assistant",
            "content": "<tts_start>",
            "eot": False,
        },  # Insert <tts_start> for speech response
    ]
    tokens, text, audio = model(messages, max_tokens=2048, temperature=0.7, do_sample=True)
    print(text)
    # print(tokens)
    audio = [x for x in audio if x < 6561]  # remove audio padding
    audio = token2wav(audio, prompt_wav="/Step-Audio2/assets/default_female.wav")
    with open(f"{ASSETS_DIR}/StepAudio2/output-paralinguistic.wav", "wb") as f:
        f.write(audio)


# Audio understanding
def instruct_mmau_test(model, token2wav):
    messages = [
        {
            "role": "system",
            "content": "You are an expert in audio analysis, please analyze the audio content and answer the questions accurately.",
        },
        {
            "role": "human",
            "content": [
                {"type": "audio", "audio": "/Step-Audio2/assets/mmau_test.wav"},
                {
                    "type": "text",
                    "text": f"Which of the following best describes the male vocal in the audio? Please choose the answer from the following options: [Soft and melodic, Aggressive and talking, High-pitched and singing, Whispering] Output the final answer in <RESPONSE> </RESPONSE>.",
                },
            ],
        },
        {"role": "assistant", "content": None},
    ]
    tokens, text, _ = model(messages, max_new_tokens=256, num_beams=2)
    print(text)


# Audio understanding
def instruct_mmau_audio_answer_test(model, token2wav):
    messages = [
        {
            "role": "system",
            "content": "You are an expert in audio analysis, please analyze the audio content and answer the questions accurately. \nPlease communicate with me via voice.\n",
        },
        {
            "role": "human",
            "content": [
                {"type": "audio", "audio": "/Step-Audio2/assets/mmau_test.wav"},
                {
                    "type": "text",
                    "text": f"Which of the following best describes the male vocal in the audio? Please choose the answer from the following options: [Soft and melodic, Aggressive and talking, High-pitched and singing, Whispering].",
                },
            ],
        },
        {
            "role": "assistant",
            "content": "<tts_start>",
            "eot": False,
        },  # Insert <tts_start> for speech response
    ]
    tokens, text, audio = model(messages, max_tokens=2048, temperature=0.7, do_sample=True)
    print(text)
    # print(tokens)
    audio = [x for x in audio if x < 6561]  # remove audio padding
    audio = token2wav(audio, prompt_wav="/Step-Audio2/assets/default_female.wav")
    with open(f"{ASSETS_DIR}/StepAudio2/output-mmau.wav", "wb") as f:
        f.write(audio)


def instruct_think_test(model, token2wav):
    history = [
        {
            "role": "system",
            "content": "你的名字叫小跃，你是由阶跃星辰(StepFun)公司训练出来的语音大模型，你能听见用户的声音特征并在思维过程中描述出来，请激活深度思考模式，通过逐步分析、逻辑推理来解决用户的问题。",
        }
    ]
    for round_idx, inp_audio in enumerate(
        [
            "/Step-Audio2/assets/multi-turn-round1-听说荡口古镇从下个月开始取消门票了，你知道这事吗。.wav",
            "/Step-Audio2/assets/multi-turn-round2-新闻说九月十九号就免费开放了。好像整个古镇都升级改造了，现在变成开放式街区了。.wav",
        ]
    ):
        print("round: ", round_idx)
        history.append({"role": "human", "content": [{"type": "audio", "audio": inp_audio}]})
        # get think content, stop when "</think>" appears
        history.append({"role": "assistant", "content": "\n<think>\n", "eot": False})
        _, think_content, _ = model(
            history, max_new_tokens=2048, temperature=0.7, do_sample=True, stop_strings=["</think>"]
        )
        print("<think>" + think_content + ">")
        # get audio response
        history[-1]["content"] += think_content + ">\n\n<tts_start>"
        print(round_idx, history)
        tokens, text, audio = model(history, max_new_tokens=2048, temperature=0.7, do_sample=True)
        print(text)
        audio = [x for x in audio if x < 6561]  # remove audio padding
        audio = token2wav(audio, prompt_wav="/Step-Audio2/assets/default_female.wav")
        with open(f"{ASSETS_DIR}/StepAudio2/output-round-{round_idx}-think.wav", "wb") as f:
            f.write(audio)
        # remove think content from history
        history.pop(-1)
        history.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "<tts_start>"},
                    {"type": "token", "token": tokens},
                ],
            }
        )


def test_instruct(gpu_prop, **kwargs):
    model = StepAudio2(MODEL_PATH)
    token2wav = Token2wav(f"{MODEL_PATH}/token2wav")

    test_func = kwargs.get("test_func", "instruct_asr_test")
    globals()[test_func](model, token2wav)


# -------------------------------------------------------------------------------------------------------------
# generate stream


# ASR (not live asr)
def stream_asr_test(model, token2wav=None):
    messages = [
        {"role": "system", "content": "请记录下你所听到的语音内容。"},
        {
            "role": "human",
            "content": [
                {
                    "type": "audio",
                    "audio": "/Step-Audio2/assets/give_me_a_brief_introduction_to_the_great_wall.wav",
                }
            ],
        },
        {"role": "assistant", "content": None},
    ]
    eos_token_id = model.llm_tokenizer.convert_tokens_to_ids("<|endoftext|>")
    token_iter = model(
        messages,
        max_new_tokens=256,
        temperature=0.1,
        do_sample=True,
        eos_token_id=[model.eos_token_id, eos_token_id],
    )
    output_text_token_ids = []
    output_audio_token_ids = []
    output_text = ""
    is_tag = False
    for token_id in token_iter:
        token = model.llm_tokenizer.decode(token_id)
        print(token_id, token)

        if token_id == 27:
            is_tag = True
            continue
        if token_id == 29:
            is_tag = False
            continue
        if is_tag:
            continue
        if token_id in [model.eos_token_id, eos_token_id]:
            break

        if token_id < 151688:
            output_text_token_ids.append(token_id)
        if token_id > 151695:
            output_audio_token_ids.append(token_id - 151696)
        output_text += token
    print(output_text)


# TTS（support: en,zh,ja)
def stream_tts_test(model, token2wav):
    messages = [
        {"role": "system", "content": "以自然的语速读出下面的文字。\n"},
        {"role": "human", "content": "你好呀，我是你的AI助手，很高兴认识你！"},
        {
            "role": "assistant",
            "content": "<tts_start>",
            "eot": False,
        },  # Insert <tts_start> for speech response
    ]
    token_iter = model(messages, max_tokens=2048, temperature=0.7, do_sample=True)
    output_text_token_ids = []
    output_audio_token_ids = []
    output_token = ""

    # stream audio
    buffer = []
    prompt_wav = "/Step-Audio2/assets/default_male.wav"
    token2wav.set_stream_cache(prompt_wav)
    output_stream = Path(f"{ASSETS_DIR}/StepAudio2/output-chunks-stream-tts.pcm")
    output_stream.unlink(missing_ok=True)
    for token_id in token_iter:
        token = model.llm_tokenizer.decode(token_id)
        print(token_id, token)
        output_token += token

        if token_id < 151688:  # text
            output_text_token_ids.append(token_id)
        if token_id > 151695:  # audio
            audio_token_id = token_id - 151696
            if audio_token_id < 6561:  # remove audio padding
                output_audio_token_ids.append(audio_token_id)
                buffer.append(audio_token_id)
                if len(buffer) >= CHUNK_SIZE + token2wav.flow.pre_lookahead_len:
                    start = time.time()
                    output = token2wav.stream(
                        buffer[: CHUNK_SIZE + token2wav.flow.pre_lookahead_len],
                        prompt_wav=prompt_wav,
                        last_chunk=False,
                    )
                    print(len(buffer), len(output), output[:50], time.time() - start)
                    with open(output_stream, "ab") as f:
                        f.write(output)
                    buffer = buffer[CHUNK_SIZE:]

    if len(buffer) > 0:
        start = time.time()
        output = token2wav.stream(buffer, prompt_wav=prompt_wav, last_chunk=True)
        print("last_chunk", len(buffer), len(output), output[:50], time.time() - start)
        with open(output_stream, "ab") as f:
            f.write(output)

    with open(output_stream, "rb") as f:
        pcm = f.read()
    wav_path = output_stream.with_suffix(".wav")
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(pcm)

    print(output_token)
    audio = token2wav(output_audio_token_ids, prompt_wav="/Step-Audio2/assets/default_male.wav")
    with open(f"{ASSETS_DIR}/StepAudio2/output-stream-tts.wav", "wb") as f:
        f.write(audio)


def stream_aqaa_test(model, token2wav):
    history = [{"role": "system", "content": "You are a helpful assistant."}]
    for round_idx, inp_audio in enumerate(
        [
            "/Step-Audio2/assets/multi-turn-round1-听说荡口古镇从下个月开始取消门票了，你知道这事吗。.wav",
            "/Step-Audio2/assets/multi-turn-round2-新闻说九月十九号就免费开放了。好像整个古镇都升级改造了，现在变成开放式街区了。.wav",
        ]
    ):
        print("round: ", round_idx)
        history.append({"role": "human", "content": [{"type": "audio", "audio": inp_audio}]})
        history.append(
            {
                "role": "assistant",
                "content": "<tts_start>",
                "eot": False,
            },  # Insert <tts_start> for speech response
        )

        token_iter = model(history, max_tokens=2048, temperature=0.7, do_sample=True)
        output_text_token_ids = []
        output_audio_token_ids = []
        output_token = ""
        output_token_ids = []

        # stream audio
        buffer = []
        prompt_wav = "/Step-Audio2/assets/default_male.wav"
        token2wav.set_stream_cache(prompt_wav)
        output_stream = Path(f"{ASSETS_DIR}/StepAudio2/output-aqaa-{round_idx}-chunks-stream.pcm")
        output_stream.unlink(missing_ok=True)
        for token_id in token_iter:
            output_token_ids.append(token_id)
            token = model.llm_tokenizer.decode(token_id)
            print(token_id, token)
            output_token += token

            if token_id < 151688:  # text
                output_text_token_ids.append(token_id)
            if token_id > 151695:  # audio
                audio_token_id = token_id - 151696
                if audio_token_id < 6561:  # remove audio padding
                    output_audio_token_ids.append(audio_token_id)
                    buffer.append(audio_token_id)
                    if len(buffer) >= CHUNK_SIZE + token2wav.flow.pre_lookahead_len:
                        start = time.time()
                        output = token2wav.stream(
                            buffer[: CHUNK_SIZE + token2wav.flow.pre_lookahead_len],
                            prompt_wav=prompt_wav,
                            last_chunk=False,
                        )
                        print(len(buffer), len(output), output[:50], time.time() - start)
                        with open(output_stream, "ab") as f:
                            f.write(output)
                        buffer = buffer[CHUNK_SIZE:]

        if len(buffer) > 0:
            start = time.time()
            output = token2wav.stream(buffer, prompt_wav=prompt_wav, last_chunk=True)
            print("last_chunk", len(buffer), len(output), output[:50], time.time() - start)
            with open(output_stream, "ab") as f:
                f.write(output)

        with open(output_stream, "rb") as f:
            pcm = f.read()
        wav_path = output_stream.with_suffix(".wav")
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(pcm)

        print(output_token)
        audio = token2wav(output_audio_token_ids, prompt_wav="/Step-Audio2/assets/default_male.wav")
        with open(f"{ASSETS_DIR}/StepAudio2/output-stream-aqaa-{round_idx}.wav", "wb") as f:
            f.write(audio)

        history.pop(-1)
        history.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "<tts_start>"},
                    {"type": "token", "token": output_token_ids},
                ],
            }
        )


def stream_think_test(model, token2wav):
    history = [
        {
            "role": "system",
            "content": "你的名字叫小跃，你是由阶跃星辰(StepFun)公司训练出来的语音大模型，你能听见用户的声音特征并在思维过程中描述出来，请激活深度思考模式，通过逐步分析、逻辑推理来解决用户的问题。",
        }
    ]
    for round_idx, inp_audio in enumerate(
        [
            "/Step-Audio2/assets/multi-turn-round1-听说荡口古镇从下个月开始取消门票了，你知道这事吗。.wav",
            "/Step-Audio2/assets/multi-turn-round2-新闻说九月十九号就免费开放了。好像整个古镇都升级改造了，现在变成开放式街区了。.wav",
            "/Step-Audio2/assets/give_me_a_brief_introduction_to_the_great_wall.wav",
        ]
    ):
        print("round: ", round_idx)
        history.append({"role": "human", "content": [{"type": "audio", "audio": inp_audio}]})
        # get think content, stop when "</think>" appears
        history.append({"role": "assistant", "content": "\n<think>\n", "eot": False})
        for i in range(2):  # think and speak
            if i > 0:
                print(round_idx, history)
            token_iter = model(
                history,
                max_new_tokens=2048,
                temperature=0.7,
                do_sample=True,
                stop_strings=["</think>"] if i == 0 else None,
            )
            output_text_token_ids = []
            output_audio_token_ids = []
            output_token = ""
            output_token_ids = []

            # stream audio
            buffer = []
            prompt_wav = "/Step-Audio2/assets/default_male.wav"
            token2wav.set_stream_cache(prompt_wav)
            output_stream = Path(
                f"{ASSETS_DIR}/StepAudio2/output-aqaa-think-{round_idx}-chunks-stream.pcm"
            )
            output_stream.unlink(missing_ok=True)
            for token_id in token_iter:
                output_token_ids.append(token_id)
                token = model.llm_tokenizer.decode(token_id)
                print(token_id, token)
                output_token += token

                if token_id < 151688:  # text
                    output_text_token_ids.append(token_id)
                if token_id > 151695:  # audio
                    audio_token_id = token_id - 151696
                    if audio_token_id < 6561:  # remove audio padding
                        output_audio_token_ids.append(audio_token_id)
                        buffer.append(audio_token_id)
                        if len(buffer) >= CHUNK_SIZE + token2wav.flow.pre_lookahead_len:
                            start = time.time()
                            output = token2wav.stream(
                                buffer[: CHUNK_SIZE + token2wav.flow.pre_lookahead_len],
                                prompt_wav=prompt_wav,
                                last_chunk=False,
                            )
                            print(len(buffer), len(output), output[:50], time.time() - start)
                            with open(output_stream, "ab") as f:
                                f.write(output)
                            buffer = buffer[CHUNK_SIZE:]

            if len(buffer) > 0:
                start = time.time()
                output = token2wav.stream(buffer, prompt_wav=prompt_wav, last_chunk=True)
                print("last_chunk", len(buffer), len(output), output[:50], time.time() - start)
                with open(output_stream, "ab") as f:
                    f.write(output)

            if os.path.isfile(output_stream):
                with open(output_stream, "rb") as f:
                    pcm = f.read()
                wav_path = output_stream.with_suffix(".wav")
                with wave.open(str(wav_path), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(24000)
                    wf.writeframes(pcm)

                print(output_token)
                audio = token2wav(
                    output_audio_token_ids, prompt_wav="/Step-Audio2/assets/default_male.wav"
                )
                with open(
                    f"{ASSETS_DIR}/StepAudio2/output-stream-aqaa-think-{round_idx}.wav", "wb"
                ) as f:
                    f.write(audio)

            if i == 0:
                think_content = model.llm_tokenizer.decode(output_text_token_ids)
                print(think_content)

                history[-1]["content"] += think_content + "<tts_start>"
            else:
                history.pop(-1)
                history.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "<tts_start>"},
                            {"type": "token", "token": output_token_ids},
                        ],
                    }
                )


def extract_function_info(tool_calls_token: str) -> tuple:
    """
    从 tool_calls_token 字符串中提取 function_name 和 function_args

    参数格式示例：
    'function\nweb_search\n{"query": "2025年8月28日 上证指数 开盘价"}'

    返回: (function_name, function_args_dict)
    """
    # 按换行符分割字符串
    parts = tool_calls_token.split("\n")

    # 验证格式是否正确
    if len(parts) < 3 or parts[0] != "function":
        raise ValueError("无效的 tool_calls_token 格式")

    # 提取函数名
    function_name = parts[1]

    try:
        # 合并剩余部分作为 JSON 字符串（处理可能的多行 JSON）
        json_str = "\n".join(parts[2:])
        function_args = json.loads(json_str)
    except json.JSONDecodeError:
        raise ValueError("无法解析 function_args JSON")

    return function_name, function_args


def stream_aqaa_tools_test(model, token2wav):
    history = [
        {
            "role": "system",
            "content": "你的名字叫做小跃，是由阶跃星辰公司训练出来的语音大模型。\n你具备调用工具解决问题的能力，你需要根据用户的需求和上下文情景，自主选择是否调用系统提供的工具来协助用户。\n你情感细腻，观察能力强，擅长分析用户的内容，并作出善解人意的回复，说话的过程中时刻注意用户的感受，富有同理心，提供多样的情绪价值。\n今天是2025年8月28日，星期五",
        },
        {
            "role": "tool_json_schemas",
            "content": '[{"type": "function", "function": {"name": "web_search", "description": "搜索工具", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "搜索关键词"}}, "required": ["query"], "additionalProperties": false}}}]',
        },
    ]
    tool_calls_token_ids = []
    search_result = ""
    for round_idx, inp_audio in enumerate(
        [
            "/Step-Audio2/assets/帮我查一下今天上证指数的开盘价是多少.wav",
            "/Step-Audio2/assets/multi-turn-round1-听说荡口古镇从下个月开始取消门票了，你知道这事吗。.wav",
            "/Step-Audio2/assets/multi-turn-round2-新闻说九月十九号就免费开放了。好像整个古镇都升级改造了，现在变成开放式街区了。.wav",
        ]
    ):
        print("round: ", round_idx)
        tool_cn = 0
        while True:
            if len(tool_calls_token_ids) > 0:
                history.append(
                    {
                        "role": "input",
                        "content": [
                            {"type": "text", "text": search_result},
                            {
                                "type": "text",
                                "text": "\n\n\n请用口语化形式总结检索结果，简短地回答用户的问题。",
                            },
                        ],
                    }
                )
                tool_cn += 1
            else:
                history.append(
                    {"role": "human", "content": [{"type": "audio", "audio": inp_audio}]}
                )
            history.append(
                {
                    "role": "assistant",
                    "content": "<tts_start>",
                    "eot": False,
                },  # Insert <tts_start> for speech response
            )

            token_iter = model(history, max_tokens=2048, temperature=0.7, do_sample=True)
            output_text_token_ids = []
            output_audio_token_ids = []
            output_token = ""
            output_token_ids = []

            # tools
            is_tool = False
            tool_calls_token_ids = []

            # stream audio
            buffer = []
            prompt_wav = "/Step-Audio2/assets/default_male.wav"
            token2wav.set_stream_cache(prompt_wav)
            output_stream = Path(
                f"{ASSETS_DIR}/StepAudio2/output-aqaa-tools-{tool_cn}-{round_idx}-chunks-stream.pcm"
            )
            output_stream.unlink(missing_ok=True)
            for token_id in token_iter:
                output_token_ids.append(token_id)
                token = model.llm_tokenizer.decode(token_id)
                print(token_id, token)
                output_token += token

                if token_id < 151688:  # text
                    if token_id == 151657:  # <tool_call>
                        is_tool = True
                        continue
                    if token_id == 151658:  # </tool_call>
                        is_tool = False
                        continue
                    if is_tool:
                        tool_calls_token_ids.append(token_id)
                        continue
                    output_text_token_ids.append(token_id)

                if token_id > 151695:  # audio
                    audio_token_id = token_id - 151696
                    if audio_token_id < 6561:  # remove audio padding
                        output_audio_token_ids.append(audio_token_id)
                        buffer.append(audio_token_id)
                        if len(buffer) >= CHUNK_SIZE + token2wav.flow.pre_lookahead_len:
                            start = time.time()
                            output = token2wav.stream(
                                buffer[: CHUNK_SIZE + token2wav.flow.pre_lookahead_len],
                                prompt_wav=prompt_wav,
                                last_chunk=False,
                            )
                            print(len(buffer), len(output), output[:50], time.time() - start)
                            with open(output_stream, "ab") as f:
                                f.write(output)
                            buffer = buffer[CHUNK_SIZE:]

            if len(buffer) > 0:
                start = time.time()
                output = token2wav.stream(buffer, prompt_wav=prompt_wav, last_chunk=True)
                print("last_chunk", len(buffer), len(output), output[:50], time.time() - start)
                with open(output_stream, "ab") as f:
                    f.write(output)

            with open(output_stream, "rb") as f:
                pcm = f.read()
            wav_path = output_stream.with_suffix(".wav")
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(pcm)

            print(f"{output_token=}")
            output_text_tokens = model.llm_tokenizer.decode(output_text_token_ids)
            print(f"{output_text_tokens=}")

            audio = token2wav(
                output_audio_token_ids, prompt_wav="/Step-Audio2/assets/default_male.wav"
            )
            with open(
                f"{ASSETS_DIR}/StepAudio2/output-stream-aqaa-tools-{tool_cn}-{round_idx}.wav", "wb"
            ) as f:
                f.write(audio)

            history.pop(-1)
            history.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "<tts_start>"},
                        {"type": "token", "token": output_token_ids},
                    ],
                }
            )

            if len(tool_calls_token_ids) == 0:
                break  # break tool call while

            tool_calls_token = model.llm_tokenizer.decode(tool_calls_token_ids)
            print(f"{tool_calls_token=}")
            function_name, function_args = extract_function_info(tool_calls_token)
            print(f"{function_name=}")
            print(f"{function_args=}")

            # mock search
            with open("/Step-Audio2/assets/search_result.txt") as f:
                search_result = f.read().strip()
                print(f"search result: {search_result}")


def generate_stream(gpu_prop, **kwargs):
    class TokenStreamer(BaseStreamer):
        def __init__(self, skip_prompt: bool = False, timeout=None):
            self.skip_prompt = skip_prompt

            # variables used in the streaming process
            self.token_queue = Queue()
            self.stop_signal = None
            self.next_tokens_are_prompt = True
            self.timeout = timeout

        def put(self, value):
            if len(value.shape) > 1 and value.shape[0] > 1:
                raise ValueError("TextStreamer only supports batch size 1")
            elif len(value.shape) > 1:
                value = value[0]

            if self.skip_prompt and self.next_tokens_are_prompt:
                self.next_tokens_are_prompt = False
                return

            for token in value.tolist():
                self.token_queue.put(token)

        def end(self):
            self.token_queue.put(self.stop_signal)

        def __iter__(self):
            return self

        def __next__(self):
            value = self.token_queue.get(timeout=self.timeout)
            if value == self.stop_signal:
                raise StopIteration()
            else:
                return value

    class StepAudio2Stream(StepAudio2):
        def __call__(self, messages: list, **kwargs):
            messages, mels = self.apply_chat_template(messages)
            print(messages)

            # Tokenize prompts
            prompt_ids = []
            for msg in messages:
                if isinstance(msg, str):
                    prompt_ids.append(
                        self.llm_tokenizer(text=msg, return_tensors="pt", padding=True)["input_ids"]
                    )
                elif isinstance(msg, list):
                    prompt_ids.append(torch.tensor([msg], dtype=torch.int32))
                else:
                    raise ValueError(f"Unsupported content type: {type(msg)}")
            prompt_ids = torch.cat(prompt_ids, dim=-1).cuda()
            attention_mask = torch.ones_like(prompt_ids)

            # mels = None if len(mels) == 0 else torch.stack(mels).cuda()
            # mel_lengths = None if mels is None else torch.tensor([mel.shape[1] - 2 for mel in mels], dtype=torch.int32, device='cuda')
            if len(mels) == 0:
                mels = None
                mel_lengths = None
            else:
                mels, mel_lengths = padding_mels(mels)
                mels = mels.cuda()
                mel_lengths = mel_lengths.cuda()

            generation_config = dict(
                max_new_tokens=2048,
                pad_token_id=self.llm_tokenizer.pad_token_id,
                eos_token_id=self.eos_token_id,
            )
            generation_config.update(kwargs)
            generation_config = GenerationConfig(**generation_config)

            streamer = TokenStreamer(skip_prompt=True)

            generation_kwargs = dict(
                input_ids=prompt_ids,
                wavs=mels,
                wav_lens=mel_lengths,
                attention_mask=attention_mask,
                generation_config=generation_config,
                streamer=streamer,
                tokenizer=self.llm_tokenizer,
            )

            thread = Thread(target=self.llm.generate, kwargs=generation_kwargs)
            thread.start()

            stop_ids = (
                [generation_config.eos_token_id]
                if isinstance(generation_config.eos_token_id, int)
                else generation_config.eos_token_id
            )
            for token_id in streamer:
                # print(token_id, end=",", flush=True)
                if token_id in stop_ids:
                    break
                yield token_id

    model = StepAudio2Stream(MODEL_PATH)

    token2wav = Token2wav(f"{MODEL_PATH}/token2wav")
    no_experimental = torch._dynamo.list_backends()
    print(f"{no_experimental=}")
    experimental = torch._dynamo.list_backends(None)
    print(f"{experimental=}")
    token2wav.flow.scatter_cuda_graph(True)

    test_func = kwargs.get("test_func", "stream_asr_test")
    globals()[test_func](model, token2wav)


async def achatbot_step_audio2_say():
    from apipeline.frames import AudioRawFrame, StartFrame, EndFrame, CancelFrame
    from achatbot.types.frames import PathAudioRawFrame

    from achatbot.cmd.bots.voice.step_audio2.helper import (
        get_step_audio2_processor,
    )
    from achatbot.types.ai_conf import AIConfig, LLMConfig

    processor = get_step_audio2_processor(
        LLMConfig(
            processor="StepAudio2TextAudioChatProcessor",
            args={
                "init_system_prompt": "",
                "prompt_wav": "/root/.achatbot/assets/default_male.wav",
                "warmup_cn": 2,
                "chat_history_size": None,
                "text_stream_out": False,
                "no_stream_sleep_time": 0.5,
                "lm_model_name_or_path": MODEL_PATH,
                "lm_gen_max_new_tokens": 64,
                "lm_gen_temperature": 0.1,
                "lm_gen_top_k": 20,
                "lm_gen_top_p": 0.95,
                "lm_gen_repetition_penalty": 1.1,
            },
        )
    )
    await processor.start(StartFrame())

    frame_iter = processor.generator_say(
        "你好, 我是Step-Audio2，很高兴认识你。", is_push_frame=False
    )
    audio = b""
    async for frame in frame_iter:
        if isinstance(frame, AudioRawFrame):
            audio += frame.audio
        print(f"say gen_frame-->", frame)
    print(f"say {processor._session.chat_history=}")
    wav_path = Path(f"{ASSETS_DIR}/StepAudio2/output-processor-chunks-stream-say.wav")
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(audio)
    await processor.stop(EndFrame())


async def achatbot_step_audio2_t2st(processor_name: str):
    from apipeline.frames import AudioRawFrame, StartFrame, EndFrame, CancelFrame, TextFrame
    from achatbot.types.frames import PathAudioRawFrame

    from achatbot.cmd.bots.voice.step_audio2.helper import (
        get_step_audio2_processor,
    )
    from achatbot.types.ai_conf import AIConfig, LLMConfig

    processor = get_step_audio2_processor(
        LLMConfig(
            processor="StepT2STProcessor",
            args={
                "init_system_prompt": "",
                "prompt_wav": "/root/.achatbot/assets/default_male.wav",
                "warmup_cn": 2,
                "chat_history_size": None,
                "text_stream_out": False,
                "no_stream_sleep_time": 0.5,
                "lm_model_name_or_path": MODEL_PATH,
                "lm_gen_max_new_tokens": 64,
                "lm_gen_temperature": 0.1,
                "lm_gen_top_k": 20,
                "lm_gen_top_p": 0.95,
                "lm_gen_repetition_penalty": 1.1,
            },
        )
    )
    await processor.start(StartFrame())

    frame_iter = processor.run_text(TextFrame(text="你好, 我是Step-Audio2，很高兴认识你。"))
    audio = b""
    async for frame in frame_iter:
        if isinstance(frame, AudioRawFrame):
            audio += frame.audio
        print(f"say gen_frame-->", frame)
    print(f"say {processor._session.chat_history=}")
    wav_path = Path(f"{ASSETS_DIR}/StepAudio2/output-processor-chunks-stream-t2st.wav")
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(audio)
    await processor.stop(EndFrame())


async def achatbot_step_audio2_audio2text(processor_name):
    from apipeline.frames import AudioRawFrame, StartFrame, EndFrame, CancelFrame
    from achatbot.types.frames import PathAudioRawFrame

    from achatbot.cmd.bots.voice.step_audio2.helper import (
        get_step_audio2_processor,
    )
    from achatbot.types.ai_conf import AIConfig, LLMConfig

    processor = get_step_audio2_processor(
        LLMConfig(
            processor=processor_name,
            args={
                "init_system_prompt": "",
                # "prompt_wav": "/root/.achatbot/assets/default_male.wav",
                "warmup_cn": 2,
                "chat_history_size": None,
                "text_stream_out": False,
                "no_stream_sleep_time": 0.5,
                "lm_model_name_or_path": MODEL_PATH,
                "lm_gen_max_new_tokens": 1024,
                "lm_gen_temperature": 0.1,
                "lm_gen_top_k": 20,
                "lm_gen_top_p": 0.9,
                "lm_gen_repetition_penalty": 1.1,
                "is_speaking": False,
            },
        )
    )
    await processor.start(StartFrame())
    for round_idx, audio_path in enumerate(
        [
            "/Step-Audio2/assets/give_me_a_brief_introduction_to_the_great_wall.wav",
            "/Step-Audio2/assets/multi-turn-round1-听说荡口古镇从下个月开始取消门票了，你知道这事吗。.wav",
            "/Step-Audio2/assets/multi-turn-round2-新闻说九月十九号就免费开放了。好像整个古镇都升级改造了，现在变成开放式街区了。.wav",
        ]
    ):
        print("round: ", round_idx)
        frame_iter = processor.run_voice(
            PathAudioRawFrame(
                path=audio_path,
                audio=b"",
            )
        )
        audio = b""
        async for frame in frame_iter:
            if isinstance(frame, AudioRawFrame):
                audio += frame.audio
            print(f"{round_idx=} gen_frame-->", frame)
        print(f"{round_idx=} {processor._session.chat_history=}")
        if len(audio) > 0:
            wav_path = Path(
                f"{ASSETS_DIR}/StepAudio2/output-{processor_name}-chunks-stream-{round_idx}.wav"
            )
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(audio)
    await processor.stop(EndFrame())


async def achatbot_step_audio2_s2st(processor_name):
    from apipeline.frames import AudioRawFrame, StartFrame, EndFrame, CancelFrame
    from achatbot.types.frames import PathAudioRawFrame

    from achatbot.cmd.bots.voice.step_audio2.helper import (
        get_step_audio2_processor,
    )
    from achatbot.types.ai_conf import AIConfig, LLMConfig

    processor_name = "StepS2STProcessor"
    processor = get_step_audio2_processor(
        LLMConfig(
            processor=processor_name,
            args={
                "init_system_prompt": "请仔细聆听这段语音，然后将其内容翻译成中文并用语音播报。",
                # "init_system_prompt": "请仔细聆听这段语音，然后将其内容翻译成英文并用语音播报。",
                "prompt_wav": "/root/.achatbot/assets/default_male.wav",
                "warmup_cn": 2,
                "chat_history_size": None,
                "text_stream_out": False,
                "no_stream_sleep_time": 0.5,
                "lm_model_name_or_path": MODEL_PATH,
                "lm_gen_max_new_tokens": 1024,
                "lm_gen_temperature": 0.7,
                "lm_gen_top_k": 20,
                "lm_gen_top_p": 0.9,
                "lm_gen_repetition_penalty": 1.1,
                "verbose": True,
            },
        )
    )
    await processor.start(StartFrame())
    for round_idx, audio_path in enumerate(
        [
            "/Step-Audio2/assets/give_me_a_brief_introduction_to_the_great_wall.wav",
            # "/Step-Audio2/assets/multi-turn-round1-听说荡口古镇从下个月开始取消门票了，你知道这事吗。.wav",
            # "/Step-Audio2/assets/multi-turn-round2-新闻说九月十九号就免费开放了。好像整个古镇都升级改造了，现在变成开放式街区了。.wav",
        ]
    ):
        print("round: ", round_idx)
        frame_iter = processor.run_voice(
            PathAudioRawFrame(
                path=audio_path,
                audio=b"",
            )
        )
        audio = b""
        async for frame in frame_iter:
            if isinstance(frame, AudioRawFrame):
                audio += frame.audio
            print(f"{round_idx=} gen_frame-->", frame)
        print(f"{round_idx=} {processor._session.chat_history=}")
        if len(audio) > 0:
            wav_path = Path(
                f"{ASSETS_DIR}/StepAudio2/output-{processor_name}-chunks-stream-{round_idx}.wav"
            )
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(audio)
    await processor.stop(EndFrame())


async def achatbot_step_audio2_aqaa(processor_name):
    from apipeline.frames import AudioRawFrame, StartFrame, EndFrame, CancelFrame
    from achatbot.types.frames import PathAudioRawFrame

    from achatbot.cmd.bots.voice.step_audio2.helper import (
        get_step_audio2_processor,
    )
    from achatbot.types.ai_conf import AIConfig, LLMConfig

    processor_name = "StepAudio2TextAudioChatProcessor"
    processor = get_step_audio2_processor(
        LLMConfig(
            processor=processor_name,
            args={
                "init_system_prompt": "",
                "prompt_wav": "/root/.achatbot/assets/default_male.wav",
                "warmup_cn": 2,
                "chat_history_size": None,
                "text_stream_out": False,
                "no_stream_sleep_time": 0.5,
                "lm_model_name_or_path": MODEL_PATH,
                "lm_gen_max_new_tokens": 1024,
                "lm_gen_temperature": 0.7,
                "lm_gen_top_k": 20,
                "lm_gen_top_p": 0.9,
                "lm_gen_repetition_penalty": 1.1,
            },
        )
    )
    await processor.start(StartFrame())
    for round_idx, audio_path in enumerate(
        [
            "/Step-Audio2/assets/multi-turn-round1-听说荡口古镇从下个月开始取消门票了，你知道这事吗。.wav",
            "/Step-Audio2/assets/multi-turn-round2-新闻说九月十九号就免费开放了。好像整个古镇都升级改造了，现在变成开放式街区了。.wav",
        ]
    ):
        print("round: ", round_idx)
        frame_iter = processor.run_voice(
            PathAudioRawFrame(
                path=audio_path,
                audio=b"",
            )
        )
        audio = b""
        async for frame in frame_iter:
            if isinstance(frame, AudioRawFrame):
                audio += frame.audio
            print(f"{round_idx=} gen_frame-->", frame)
        print(f"{round_idx=} {processor._session.chat_history=}")
        if len(audio) > 0:
            wav_path = Path(
                f"{ASSETS_DIR}/StepAudio2/output-{processor_name}-chunks-stream-{round_idx}.wav"
            )
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(audio)
    await processor.stop(EndFrame())


async def achatbot_step_audio2_aqaa_tools(processor_name):
    from apipeline.frames import AudioRawFrame, StartFrame, EndFrame, CancelFrame
    from achatbot.types.frames import PathAudioRawFrame, FunctionCallFrame

    from achatbot.cmd.bots.voice.step_audio2.helper import (
        get_step_audio2_processor,
    )
    from achatbot.types.ai_conf import AIConfig, LLMConfig

    processor_name = "StepAudio2TextAudioChatProcessor"
    processor = get_step_audio2_processor(
        LLMConfig(
            processor=processor_name,
            args={
                "init_system_prompt": "你的名字叫做小跃，是由阶跃星辰公司训练出来的语音大模型。\n你具备调用工具解决问题的能力，你需要根据用户的需求和上下文情景，自主选择是否调用系统提供的工具来协助用户。\n你情感细腻，观察能力强，擅长分析用户的内容，并作出善解人意的回复，说话的过程中时刻注意用户的感受，富有同理心，提供多样的情绪价值。\n今天是2025年9月12日，星期五",
                "prompt_wav": "/root/.achatbot/assets/default_male.wav",
                "warmup_cn": 2,
                "chat_history_size": None,
                "text_stream_out": False,
                "no_stream_sleep_time": 0.5,
                # "tools": ["web_search","get_weather"],
                "tools": ["web_search"],
                "lm_model_name_or_path": MODEL_PATH,
                "lm_gen_max_new_tokens": 1024,
                "lm_gen_temperature": 0.7,
                "lm_gen_top_k": 20,
                "lm_gen_top_p": 0.9,
                "lm_gen_repetition_penalty": 1.1,
                "verbose": True,
            },
        ),
    )
    print(f"{processor.chat_history=}")
    await processor.start(StartFrame())
    for round_idx, audio_path in enumerate(
        [
            "/Step-Audio2/assets/帮我查一下今天上证指数的开盘价是多少.wav",
            # "/Step-Audio2/assets/multi-turn-round1-听说荡口古镇从下个月开始取消门票了，你知道这事吗。.wav",
            # "/Step-Audio2/assets/multi-turn-round2-新闻说九月十九号就免费开放了。好像整个古镇都升级改造了，现在变成开放式街区了。.wav",
        ]
    ):
        print("round: ", round_idx)
        frame_iter = processor.run_voice(
            PathAudioRawFrame(
                path=audio_path,
                audio=b"",
            )
        )
        audio = b""
        tool_cn = 0
        async for frame in frame_iter:
            if isinstance(frame, AudioRawFrame):
                audio += frame.audio
            if isinstance(frame, FunctionCallFrame):
                tool_cn += 1
            print(f"{round_idx=} gen_frame-->", frame)
        print(f"{round_idx=} {processor.chat_history=}")
        if len(audio) > 0:
            wav_path = Path(
                f"{ASSETS_DIR}/StepAudio2/output-{processor_name}-tools-chunks-stream-{round_idx}.wav"
            )
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(audio)
    await processor.stop(EndFrame())


async def achatbot_step_audio2_aqaa_think(processor_name):
    from apipeline.frames import AudioRawFrame, StartFrame, EndFrame, CancelFrame
    from achatbot.types.frames import PathAudioRawFrame, FunctionCallFrame, ReasoningThinkTextFrame

    from achatbot.cmd.bots.voice.step_audio2.helper import (
        get_step_audio2_processor,
    )
    from achatbot.types.ai_conf import AIConfig, LLMConfig

    processor_name = "StepAudio2TextAudioThinkChatProcessor"
    processor = get_step_audio2_processor(
        LLMConfig(
            processor=processor_name,
            args={
                "init_system_prompt": "你的名字叫小跃，你是由阶跃星辰(StepFun)公司训练出来的语音大模型，你能听见用户的声音特征并在思维过程中描述出来，请激活深度思考模式，通过逐步分析、逻辑推理来解决用户的问题。",
                "prompt_wav": "/root/.achatbot/assets/default_male.wav",
                "warmup_cn": 2,
                "chat_history_size": None,
                "text_stream_out": False,
                "no_stream_sleep_time": 0.5,
                "is_reasoning_think": True,
                # "tools": ["web_search"],
                "lm_model_name_or_path": MODEL_PATH,
                "lm_gen_max_new_tokens": 10240,
                "lm_gen_temperature": 0.7,
                "lm_gen_top_k": 20,
                "lm_gen_top_p": 0.9,
                "lm_gen_repetition_penalty": 1.1,
                "lm_gen_stop_ids": [151665],
                "verbose": True,
            },
        ),
    )
    print(f"{processor.chat_history=}")
    await processor.start(StartFrame())
    for round_idx, audio_path in enumerate(
        [
            # "/Step-Audio2/assets/帮我查一下今天上证指数的开盘价是多少.wav",
            # "/Step-Audio2/assets/multi-turn-round1-听说荡口古镇从下个月开始取消门票了，你知道这事吗。.wav",
            # "/Step-Audio2/assets/multi-turn-round2-新闻说九月十九号就免费开放了。好像整个古镇都升级改造了，现在变成开放式街区了。.wav",
            "/Step-Audio2/assets/give_me_a_brief_introduction_to_the_great_wall.wav",
        ]
    ):
        print("round: ", round_idx)
        frame_iter = processor.run_voice(
            PathAudioRawFrame(
                path=audio_path,
                audio=b"",
            )
        )
        audio = b""
        tool_cn = 0
        think_cn = 0
        async for frame in frame_iter:
            if isinstance(frame, AudioRawFrame):
                audio += frame.audio
            if isinstance(frame, FunctionCallFrame):
                tool_cn += 1
            if isinstance(frame, ReasoningThinkTextFrame):
                think_cn += 1
            print(f"{round_idx=} gen_frame-->{str(frame)}")
        print(f"{round_idx=} {think_cn=} {processor.chat_history=}")
        if len(audio) > 0:
            wav_path = Path(
                f"{ASSETS_DIR}/StepAudio2/output-{processor_name}-think-chunks-stream-{round_idx}.wav"
            )
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(audio)
    await processor.stop(EndFrame())


async def achatbot_step_audio2_processor(gpu_prop, **kwargs):
    from achatbot.common.logger import Logger

    Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)

    test_func = kwargs.get("test_func", "achatbot_step_audio2_aqaa")
    processor_name = kwargs.get("processor_name") or "StepASRProcessor"
    await globals()[test_func](processor_name)


"""
modal run src/download_models.py --repo-ids "stepfun-ai/Step-Audio-2-mini-Base"
modal run src/download_models.py --repo-ids "stepfun-ai/Step-Audio-2-mini"
modal run src/download_models.py --repo-ids "stepfun-ai/Step-Audio-2-mini-Think"

IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task dump_model
IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task tokenize
LLM_MODEL=stepfun-ai/Step-Audio-2-mini-Base IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task tokenize
LLM_MODEL=stepfun-ai/Step-Audio-2-mini-Think IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task tokenize

IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task test_base --test-func asr_test
IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task test_base --test-func audio_caption_test
IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task test_base --test-func tts_test
IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task test_base --test-func s2st_test
IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task test_base --test-func t2st_test
IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task test_base --test-func multi_turn_aqta_test
IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task test_base --test-func multi_turn_aqaa_test

IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task test_instruct --test-func instruct_asr_test
IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task test_instruct --test-func instruct_audio_caption_test
IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task test_instruct --test-func instruct_s2tt_test
IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task test_instruct --test-func instruct_s2st_test
IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task test_instruct --test-func instruct_multi_turn_tqta_test
IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task test_instruct --test-func instruct_multi_turn_tqaa_test
IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task test_instruct --test-func instruct_multi_turn_aqta_test
IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task test_instruct --test-func instruct_multi_turn_aqaa_test
IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task test_instruct --test-func instruct_tool_call_test
IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task test_instruct --test-func instruct_paralinguistic_test
IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task test_instruct --test-func instruct_mmau_test
IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task test_instruct --test-func instruct_mmau_audio_answer_test
IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task test_instruct --test-func instruct_think_test

IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task generate_stream --test-func stream_asr_test
IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task generate_stream --test-func stream_tts_test
IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task generate_stream --test-func stream_aqaa_test 
IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task generate_stream --test-func stream_aqaa_tools_test
IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task generate_stream --test-func stream_think_test

IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task achatbot_step_audio2_processor --test-func=achatbot_step_audio2_audio2text --processor-name=StepASRProcessor
IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task achatbot_step_audio2_processor --test-func=achatbot_step_audio2_audio2text --processor-name=StepAudioCaptionProcessor
IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task achatbot_step_audio2_processor --test-func=achatbot_step_audio2_audio2text --processor-name=StepS2TTProcessor
IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task achatbot_step_audio2_processor --test-func=achatbot_step_audio2_say
IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task achatbot_step_audio2_processor --test-func=achatbot_step_audio2_t2st
IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task achatbot_step_audio2_processor --test-func=achatbot_step_audio2_s2st
IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task achatbot_step_audio2_processor --test-func=achatbot_step_audio2_aqaa
IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task achatbot_step_audio2_processor --test-func=achatbot_step_audio2_aqaa_tools
IMAGE_GPU=L4 modal run src/llm/transformers/step_audio2.py --task achatbot_step_audio2_processor --test-func=achatbot_step_audio2_aqaa_think
IMAGE_GPU=L4 LLM_MODEL=stepfun-ai/Step-Audio-2-mini-Think modal run src/llm/transformers/step_audio2.py --task achatbot_step_audio2_processor --test-func=achatbot_step_audio2_aqaa_think
"""


@app.local_entrypoint()
def main(task: str = "dump_model", test_func="", processor_name=""):
    tasks = {
        "dump_model": dump_model,
        "tokenize": tokenize,
        "test_base": test_base,
        "test_instruct": test_instruct,
        "generate_stream": generate_stream,
        "achatbot_step_audio2_processor": achatbot_step_audio2_processor,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(
        tasks[task],
        test_func=test_func,
        processor_name=processor_name,
    )
