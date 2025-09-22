import os
import base64
import json
import time
import re
import io
import wave
import threading


import modal
import torchaudio
import requests


vllm_image = (
    modal.Image.from_registry(
        "stepfun2025/vllm:step-audio-2-v20250909",
        add_python="3.12",
    )
    .pip_install("requests", "torchaudio", "numpy")
    .env(
        {
            "LLM_MODEL": os.getenv("LLM_MODEL", "stepfun-ai/Step-Audio-2-mini"),
            "LLM_MODEL_NAME": os.getenv("LLM_MODEL_NAME", "step-audio-2-mini"),  # mini, mini-think
            "VLLM_USE_V1": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCH_CUDA_ARCH_LIST": "7.5 8.0 8.6 8.7 8.9 9.0 10.0",
        }
    )
)


app = modal.App("vllm-step-audio2")
IMAGE_GPU = os.getenv("IMAGE_GPU", None)  # L4 L40s L4:2
# how many requests can one replica handle? tune carefully!
CONCURRENT_MAX_INPUTS = int(os.getenv("CONCURRENT_MAX_INPUTS", 1))
MAX_CONTAINER_COUNT = int(os.getenv("MAX_CONTAINER_COUNT", 1))

MINUTES = 60  # seconds
VLLM_PORT = 8000

HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
VLLM_CACHE_DIR = "/root/.cache/vllm"
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_vol = modal.Volume.from_name("assets", create_if_missing=True)

LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "../../models/stepfun-ai/Step-Audio-2-mini")


def load_audio(file_path, target_rate=16000, max_length=None):
    """
    Open an audio file and read as mono waveform, resampling as necessary
    If max_length is provided, truncate the audio to that length
    """
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != target_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_rate)(
            waveform
        )
    audio = waveform[0]  # get the first channel

    # Truncate audio if it exceeds max_length
    if max_length is not None and audio.shape[0] > max_length:
        audio = audio[:max_length]

    return audio


class StepAudio2:
    audio_token_re = re.compile(r"<audio_(\d+)>")

    def __init__(self, api_url, model_name, tokenizer_path: str = None):
        self.api_url = api_url
        self.model_name = model_name

        from transformers import AutoTokenizer

        self.llm_tokenizer = None
        if tokenizer_path:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, trust_remote_code=True, padding_side="right"
            )

    def __call__(self, messages, **kwargs):
        return next(self.stream(messages, **kwargs, stream=False))

    def stream(self, messages, stream=True, **kwargs):
        headers = {"Content-Type": "application/json"}
        payload = kwargs
        payload["messages"] = self.apply_chat_template(messages)
        payload["model"] = self.model_name
        payload["stream"] = stream
        if (payload["messages"][-1].get("role", None) == "assistant") and (
            payload["messages"][-1].get("content", None) is None
        ):
            payload["messages"].pop(-1)
            payload["continue_final_message"] = False
            payload["add_generation_prompt"] = True
        elif payload["messages"][-1].get("eot", True):
            payload["continue_final_message"] = False
            payload["add_generation_prompt"] = True
        else:
            payload["continue_final_message"] = True
            payload["add_generation_prompt"] = False
        with requests.post(self.api_url, headers=headers, json=payload, stream=stream) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                print(line)
                if line == b"":
                    continue
                line = line.decode("utf-8")[6:] if stream else line.decode("utf-8")
                if line == "[DONE]":
                    continue
                line = json.loads(line)["choices"][0]["delta" if stream else "message"]
                text = line.get("tts_content", {}).get("tts_text", None)
                text = text if text else line["content"]
                audio = line.get("tts_content", {}).get("tts_audio", None)
                audio_id = (
                    [int(i) for i in StepAudio2.audio_token_re.findall(audio)] if audio else None
                )

                token_ids = None
                if text and self.llm_tokenizer:
                    token_ids = self.llm_tokenizer.encode(text)
                elif audio and self.llm_tokenizer:
                    token_ids = self.llm_tokenizer.encode(audio)

                yield (line, text, audio_id, token_ids)

    def process_content_item(self, item):
        if item["type"] == "audio":
            audio_tensor = load_audio(item["audio"], target_rate=16000)
            chunks = []
            for i in range(0, audio_tensor.shape[0], 25 * 16000):
                chunk = audio_tensor[i : i + 25 * 16000]
                if len(chunk.numpy()) == 0:
                    continue
                chunk_int16 = (chunk.numpy().clip(-1.0, 1.0) * 32767.0).astype("int16")
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(chunk_int16.tobytes())
                chunks.append(
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(buf.getvalue()).decode("utf-8"),
                            "format": "wav",
                        },
                    }
                )
            return chunks
        return [item]

    def apply_chat_template(self, messages):
        out = []
        for m in messages:
            if m["role"] == "human" and isinstance(m["content"], list):
                out.append(
                    {
                        "role": m["role"],
                        "content": [j for i in m["content"] for j in self.process_content_item(i)],
                    }
                )
            else:
                out.append(m)
        return out


@app.function(
    image=vllm_image,
    gpu=IMAGE_GPU,
    max_containers=MAX_CONTAINER_COUNT,
    # how long should we stay up with no requests?
    scaledown_window=30 * MINUTES,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_vol,
        VLLM_CACHE_DIR: vllm_cache_vol,
    },
)
# how many requests can one replica handle? tune carefully!
@modal.concurrent(max_inputs=CONCURRENT_MAX_INPUTS)
@modal.web_server(port=VLLM_PORT, startup_timeout=5 * MINUTES)
def serve():
    import subprocess

    MODEL_ID = os.getenv("LLM_MODEL", "stepfun-ai/Step-Audio-2-mini")
    model_path = os.path.join(
        HF_MODEL_DIR,
        MODEL_ID,
    )
    MODEL_NAME = os.getenv("LLM_MODEL_NAME", "step-audio-2-mini")
    # subprocess.run(f"vllm serve -h", shell=True)
    subprocess.run(f"which vllm", shell=True)
    subprocess.run(f"vllm --version", shell=True)
    cmd = [
        "vllm",
        "serve",
        model_path,
        "--uvicorn-log-level=info",
        "--served-model-name",
        MODEL_NAME,
        "--port",
        str(VLLM_PORT),
        "--max-model-len",
        "16384",
        "--max-num-seqs",
        "32",
        "--tensor-parallel-size",
        "1",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "step_audio_2",
        "--tokenizer-mode",
        "step_audio_2",
        "--chat_template_content_format",
        "string",
        "--audio-parser",
        "step_audio_2_tts_ta4",
        "--trust-remote-code",
    ]

    # Refactored to avoid shell=True for better security
    print(" ".join(cmd))
    subprocess.Popen(cmd)


def test_request(model: StepAudio2, token2wav):
    # Text-to-speech conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "human", "content": "Give me a brief introduction to the Great Wall."},
        {
            "role": "assistant",
            "content": "<tts_start>",
            "eot": False,
        },  # Insert <tts_start> for speech response
    ]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "human",
            "content": [
                {
                    "type": "audio",
                    "audio": "../../deps/StepAudio2/assets/give_me_a_brief_introduction_to_the_great_wall.wav",
                }
            ],
        },
        {
            "role": "assistant",
            "content": "<tts_start>",
            "eot": False,
        },  # Insert <tts_start> for speech response
    ]
    payload = {}
    payload["messages"] = model.apply_chat_template(messages)
    payload["model"] = model.model_name
    payload["stream"] = True
    if (payload["messages"][-1].get("role", None) == "assistant") and (
        payload["messages"][-1].get("content", None) is None
    ):
        payload["messages"].pop(-1)
        payload["continue_final_message"] = False
        payload["add_generation_prompt"] = True
    elif payload["messages"][-1].get("eot", True):
        payload["continue_final_message"] = False
        payload["add_generation_prompt"] = True
    else:
        payload["continue_final_message"] = True
        payload["add_generation_prompt"] = False
    print(payload)


def test_text2text(model, token2wav=None):
    # Text-to-text conversation
    sampling_params = {
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
        "frequency_penalty": 0,
        "repetition_penalty": 1.05,
        "skip_special_tokens": False,
        "parallel_tool_calls": False,
    }
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "human", "content": "Give me a brief introduction to the Great Wall."},
        {"role": "assistant", "content": None},
    ]
    response, text, _, token_ids = model(messages, **sampling_params)
    print(text)
    print(f"{token_ids=}")


def test_text2text_stream(model: StepAudio2, token2wav=None):
    # Text-to-text conversation
    sampling_params = {
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
        "frequency_penalty": 0,
        "repetition_penalty": 1.05,
        "skip_special_tokens": False,
        "parallel_tool_calls": False,
        "logprobs": True,
    }
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "human", "content": "Give me a brief introduction to the Great Wall."},
        {"role": "assistant", "content": None},
    ]
    stream_iter = model.stream(messages, stream=True, **sampling_params)
    for response, text, _, token_ids in stream_iter:
        print(f"{response=} {text=} {token_ids=}")


def test_text2speech_stream(model: StepAudio2, token2wav):
    # Text-to-speech conversation
    messages = [
        {"role": "system", "content": "以自然的语速读出下面的文字。\n"},
        {"role": "human", "content": "Give me a brief introduction to the Great Wall."},
        {
            "role": "assistant",
            "content": "<tts_start>",
            "eot": False,
        },  # Insert <tts_start> for speech response
    ]
    sampling_params = {
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
        "frequency_penalty": 0,
        "repetition_penalty": 1.05,
        "skip_special_tokens": False,
        "parallel_tool_calls": False,
        "return_token_ids": True,
    }
    stream_iter = model.stream(messages, stream=True, **sampling_params)
    for response, text, audio, token_id in stream_iter:
        print(f"{response=} {text=} {audio=} {token_id=}")


def test_text2speech(model: StepAudio2, token2wav):
    # Text-to-speech conversation
    messages = [
        {"role": "system", "content": "以自然的语速读出下面的文字。\n"},
        {"role": "human", "content": "Give me a brief introduction to the Great Wall."},
        {
            "role": "assistant",
            "content": "<tts_start>",
            "eot": False,
        },  # Insert <tts_start> for speech response
    ]
    sampling_params = {
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
        "frequency_penalty": 0,
        "repetition_penalty": 1.05,
        "skip_special_tokens": False,
        "parallel_tool_calls": False,
    }
    response, text, audio, token_ids = model(messages, **sampling_params)
    print(text)
    print(audio)
    print(f"{token_ids=}")
    audio = token2wav(audio, prompt_wav="../../deps/StepAudio2/assets/default_male.wav")
    with open("../../records/output-male.wav", "wb") as f:
        f.write(audio)


def test_speech2text(model, token2wav):
    # Speech-to-text conversation
    sampling_params = {
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
        "frequency_penalty": 0,
        "repetition_penalty": 1.05,
        "skip_special_tokens": False,
        "parallel_tool_calls": False,
    }
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "human",
            "content": [
                {
                    "type": "audio",
                    "audio": "../../deps/StepAudio2/assets/give_me_a_brief_introduction_to_the_great_wall.wav",
                }
            ],
        },
        {"role": "assistant", "content": None},
    ]
    response, text, _, token_ids = model(messages, **sampling_params)
    print(text)
    print(f"{token_ids=}")


def test_speech2speech(model: StepAudio2, token2wav):
    # Speech-to-text conversation
    sampling_params = {
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
        "frequency_penalty": 0,
        "repetition_penalty": 1.05,
        "skip_special_tokens": False,
        "parallel_tool_calls": False,
    }
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "human",
            "content": [
                {
                    "type": "audio",
                    "audio": "../../deps/StepAudio2/assets/give_me_a_brief_introduction_to_the_great_wall.wav",
                }
            ],
        },
        {
            "role": "assistant",
            "content": "<tts_start>",
            "eot": False,
        },  # Insert <tts_start> for speech response
    ]
    response, text, audio, token_ids = model(messages, **sampling_params)
    print(text)
    print(audio)
    print(f"{token_ids=}")
    audio = token2wav(audio, prompt_wav="../../deps/StepAudio2/assets/default_female.wav")
    with open("../../records/output-female.wav", "wb") as f:
        f.write(audio)


def test_speech2speech_think(model: StepAudio2, token2wav):
    # Speech-to-text conversation
    sampling_params = {
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
        "frequency_penalty": 0,
        "repetition_penalty": 1.05,
        "skip_special_tokens": False,
        "parallel_tool_calls": False,
        "stop": ["</think>"],
    }
    messages = [
        {
            "role": "system",
            "content": "你的名字叫小跃，你是由阶跃星辰(StepFun)公司训练出来的语音大模型，你能听见用户的声音特征并在思维过程中描述出来，请激活深度思考模式，通过逐步分析、逻辑推理来解决用户的问题。",
        },
        {
            "role": "human",
            "content": [
                {
                    "type": "audio",
                    "audio": "../../deps/StepAudio2/assets/give_me_a_brief_introduction_to_the_great_wall.wav",
                }
            ],
        },
        {"role": "assistant", "content": "<think>", "eot": False},
    ]
    _, think_content, _, _ = model(messages, **sampling_params)
    print("<think>" + think_content + "</think>")
    messages[-1]["content"] += think_content + "</think>" + "\n\n<tts_start>"
    response, text, audio, _ = model(
        messages, max_tokens=2048, temperature=0.7, repetition_penalty=1.05
    )
    print(text)
    print(audio)
    audio = token2wav(audio, prompt_wav="../../deps/StepAudio2/assets/default_female.wav")
    with open("../../records/output-female-think.wav", "wb") as f:
        f.write(audio)


def test_speech2speech_think_stream(model: StepAudio2, token2wav):
    # Speech-to-text conversation
    sampling_params = {
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
        "frequency_penalty": 0,
        "repetition_penalty": 1.05,
        "skip_special_tokens": False,
        "parallel_tool_calls": False,
        "stop": ["</think>"],
    }
    messages = [
        {
            "role": "system",
            "content": "你的名字叫小跃，你是由阶跃星辰(StepFun)公司训练出来的语音大模型，你能听见用户的声音特征并在思维过程中描述出来，请激活深度思考模式，通过逐步分析、逻辑推理来解决用户的问题。",
        },
        {
            "role": "human",
            "content": [
                {
                    "type": "audio",
                    "audio": "../../deps/StepAudio2/assets/give_me_a_brief_introduction_to_the_great_wall.wav",
                }
            ],
        },
        {"role": "assistant", "content": "<think>", "eot": False},
    ]
    _, think_content, _, _ = model(messages, **sampling_params)
    print("<think>" + think_content + "</think>")
    messages[-1]["content"] += think_content + "</think>" + "\n\n<tts_start>"
    # sampling_params = {
    #    "max_tokens": 1024,
    #    "temperature": 0.7,
    #    "top_p": 0.9,
    #    "frequency_penalty": 0,
    #    "repetition_penalty": 1.05,
    #    "skip_special_tokens": False,
    #    "parallel_tool_calls": False,
    #    "return_token_ids": True,
    # }
    print(messages)
    stream_iter = model.stream(messages, stream=True, **sampling_params)
    audio_tokens = []
    texts = ""
    repeat_cn = 0
    for response, text, audio, token_id in stream_iter:
        if token_id:
            repeat_cn = 0
            print(f"{response=} {text=} {audio=} {token_id=}")
        elif response["role"] is None:
            print(f"{response=} {text=} {audio=} {token_id=}")
            repeat_cn += 1
            if repeat_cn > 10:
                break
        if audio:
            audio_tokens += audio
        if text:
            texts += text
    print(texts)
    audio_bytes = token2wav(
        audio_tokens, prompt_wav="../../deps/StepAudio2/assets/default_female.wav"
    )
    with open("../../records/output-female-think-stream.wav", "wb") as f:
        f.write(audio_bytes)


def test_all(model, token2wav):
    """sequential test"""
    test_text2text(model, token2wav)
    test_text2speech(model, token2wav)
    test_speech2text(model, token2wav)
    test_speech2speech(model, token2wav)
    test_speech2speech_think(model, token2wav)


def test_all_concurrent(model, token2wav):
    """concurrent test"""
    # Create a list of all test functions to run
    tests = [
        test_text2text,
        test_text2speech,
        test_speech2text,
        test_speech2speech,
        test_speech2speech_think,
    ]

    threads = []
    # Create and start a new thread for each test function
    for test_func in tests:
        # The 'args' tuple passes arguments to the function run in the thread
        thread = threading.Thread(target=test_func, args=(model, token2wav))
        threads.append(thread)
        thread.start()  # Start the thread's activity

    # Wait for all threads to complete before moving on
    for thread in threads:
        thread.join()

    print("All concurrent tests have finished.")


def test_tokenizer(model, token2wav):
    """
    huggingface-cli download stepfun-ai/Step-Audio-2-mini --include "*.json" --local-dir ./models/stepfun-ai/Step-Audio-2-mini

    - map vocabulary to token is fast
    """
    from transformers import AutoTokenizer

    llm_tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_PATH,
        trust_remote_code=True,
        padding_side="right",
        use_fast=True,
    )
    start = time.time()
    tokens = "Give <audio_1466><audio_2112><audio_168><audio_2834> me"
    token_ids = llm_tokenizer.encode(tokens)
    print(f"{time.time() - start=}")
    print(token_ids)
    decoded_tokens = llm_tokenizer.decode(token_ids)
    assert decoded_tokens == tokens


token2wav_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("requests", "torchaudio")
    .pip_install("achatbot==0.0.25.post2")
    .env(
        {
            "ACHATBOT_PKG": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)


@app.function(
    image=token2wav_image,
    gpu=os.getenv("TOKEN2WAV_IMAGE_GPU", None),
    max_containers=1,
    # how long should we stay up with no requests?
    scaledown_window=15 * MINUTES,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_vol,
        VLLM_CACHE_DIR: vllm_cache_vol,
    },
)
def run_step_audio2_speaking():
    """use modal gpu accelerate for step-audio2 speaking(token2wav)"""
    from achatbot.processors.voice.step_audio2_processor import Token2wav

    pass


"""
# - run serve and local run cli test
IMAGE_GPU=L4 modal serve src/llm/vllm/step_audio2.py
IMAGE_GPU=L40s modal serve src/llm/vllm/step_audio2.py
LLM_MODEL_NAME=step-audio-2-mini-think LLM_MODEL=stepfun-ai/Step-Audio-2-mini-Think IMAGE_GPU=L40s modal serve src/llm/vllm/step_audio2.py
# cold start vllm serve
curl -v -XGET "https://weedge--vllm-step-audio2-serve-dev.modal.run/health"

# run step-audio2 vllm client test
cd deps/StepAudio2 && python stepaudio2vllm.py


# - run serve and test together, for CI test
ACHATBOT_PKG=1 IMAGE_GPU=L4 modal run src/llm/vllm/step_audio2.py --test test_all
ACHATBOT_PKG=1 IMAGE_GPU=L40s modal run src/llm/vllm/step_audio2.py --test test_all
ACHATBOT_PKG=1 IMAGE_GPU=L40s modal run src/llm/vllm/step_audio2.py --test test_text2text
ACHATBOT_PKG=1 IMAGE_GPU=L40s modal run src/llm/vllm/step_audio2.py --test test_text2speech
ACHATBOT_PKG=1 IMAGE_GPU=L40s modal run src/llm/vllm/step_audio2.py --test test_speech2text
ACHATBOT_PKG=1 IMAGE_GPU=L40s modal run src/llm/vllm/step_audio2.py --test test_speech2speech
LLM_MODEL=stepfun-ai/Step-Audio-2-mini-Think ACHATBOT_PKG=1 IMAGE_GPU=L40s modal run src/llm/vllm/step_audio2.py --test test_speech2speech_think

# - run serve and test streaming
ACHATBOT_PKG=1 IMAGE_GPU=L40s modal run src/llm/vllm/step_audio2.py --test test_text2text_stream
ACHATBOT_PKG=1 IMAGE_GPU=L40s modal run src/llm/vllm/step_audio2.py --test test_text2speech_stream
LLM_MODEL=stepfun-ai/Step-Audio-2-mini-Think ACHATBOT_PKG=1 IMAGE_GPU=L40s modal run src/llm/vllm/step_audio2.py --test test_speech2speech_think_stream
LLM_MODEL_NAME=step-audio-2-mini-think LLM_MODEL=stepfun-ai/Step-Audio-2-mini-Think ACHATBOT_PKG=1 IMAGE_GPU=L40s modal run src/llm/vllm/step_audio2.py --test test_speech2speech_think_stream


# audio-llm use Vllm serving on GPU(one container), token2wav run local cpu test, concurrent test (Overhead bound test)
MAX_CONTAINER_COUNT=1 CONCURRENT_MAX_INPUTS=1 ACHATBOT_PKG=1 IMAGE_GPU=L40s modal run src/llm/vllm/step_audio2.py --test test_all_concurrent
MAX_CONTAINER_COUNT=1 CONCURRENT_MAX_INPUTS=2 ACHATBOT_PKG=1 IMAGE_GPU=L40s modal run src/llm/vllm/step_audio2.py --test test_all_concurrent
MAX_CONTAINER_COUNT=1 CONCURRENT_MAX_INPUTS=4 ACHATBOT_PKG=1 IMAGE_GPU=L40s modal run src/llm/vllm/step_audio2.py --test test_all_concurrent
"""


@app.local_entrypoint()
def main(test: str = "test_all"):
    import subprocess
    from achatbot.processors.voice.step_audio2_processor import Token2wav
    from achatbot.core.llm.vllm.client.step_audio2_mini_vllm import StepAudio2MiniVLLMClient

    serve_url = serve.get_web_url()
    print(f"VLLM serving at {serve_url}/v1/chat/completions")

    # cold start vllm serve
    if test not in ["test_request", "test_tokenizer"]:
        subprocess.run(f"curl -v -XGET '{serve_url}/health'", shell=True)
        print("VLLM health check passed")

    # https://platform.openai.com/docs/api-reference/chat/create
    # model = StepAudio2(
    model = StepAudio2MiniVLLMClient(
        f"{serve_url}/v1/chat/completions",
        os.getenv("LLM_MODEL_NAME", "step-audio-2-mini"),
        tokenizer_path=LOCAL_MODEL_PATH,
    )

    token2wav = None
    if test not in ["test_text2text", "test_request"]:
        token2wav = Token2wav(f"{LOCAL_MODEL_PATH}/token2wav")
    test_map = {
        "test_all": test_all,
        "test_text2text": test_text2text,
        "test_text2speech": test_text2speech,
        "test_speech2text": test_speech2text,
        "test_speech2speech": test_speech2speech,
        "test_speech2speech_think": test_speech2speech_think,
        "test_all_concurrent": test_all_concurrent,
        "test_request": test_request,
        "test_tokenizer": test_tokenizer,
        "test_text2text_stream": test_text2text_stream,
        "test_text2speech_stream": test_text2speech_stream,
        "test_speech2speech_think_stream": test_speech2speech_think_stream,
    }
    if test in test_map:
        test_map[test](model, token2wav)
    else:
        raise ValueError(f"Unknown test: {test}")


"""
ACHATBOT_PKG=1 LLM_MODEL_NAME=step-audio-2-mini-think python src/llm/vllm/step_audio2.py
"""
if __name__ == "__main__":
    from achatbot.processors.voice.step_audio2_processor import Token2wav
    from achatbot.core.llm.vllm.client.step_audio2_mini_vllm import StepAudio2MiniVLLMClient

    model = StepAudio2MiniVLLMClient(
        # f"{serve_url}/v1/chat/completions",
        f"https://weege009--vllm-step-audio2-serve-dev.modal.run/v1/chat/completions",
        os.getenv("LLM_MODEL_NAME", "step-audio-2-mini"),
        tokenizer_path=LOCAL_MODEL_PATH,
    )
    token2wav = Token2wav(f"{LOCAL_MODEL_PATH}/token2wav")
    test_speech2speech_think_stream(model=model, token2wav=token2wav)
