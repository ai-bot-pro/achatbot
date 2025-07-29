import base64
import sys
import os
import time
import subprocess
import asyncio
import threading
from typing import List, Optional

import modal

APP_NAME = os.getenv("APP_NAME", "")

app = modal.App("higgs-audio-llm")
img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
    modal.Image.from_registry(
        "nvcr.io/nvidia/pytorch:25.02-py3",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "clang", "cmake", "ninja-build")
    .run_commands(
        "git clone https://github.com/weedge/higgs-audio.git",
        "cd /higgs-audio && pip install -r requirements.txt",
    )
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    .pip_install("soundfile")
    .run_commands(
        "cd /higgs-audio && git pull origin feat/achatbot",
        "cd /higgs-audio && git checkout b026c052bd2b6259947c1248f1cf2f14810f395e",
        # "cd /higgs-audio && git pull origin feat/stream",
        # "cd /higgs-audio && git checkout 49984bcc38a0b25e140b49cf2eb087f738f52b8a",
        "cd /higgs-audio && pip install -e .",
    )
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            # "TQDM_DISABLE": "1",
            "LLM_MODEL": os.getenv("LLM_MODEL", "bosonai/higgs-audio-v2-generation-3B-base"),
            "AUDIO_TOKENIZER_PATH": os.getenv(
                "AUDIO_TOKENIZER_PATH", "bosonai/higgs-audio-v2-tokenizer"
            ),
        }
    )
)

if APP_NAME == "achatbot":
    img = img.pip_install(
        f"achatbot==0.0.23",
        extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
    ).env(
        {
            "ACHATBOT_PKG": "1",
        }
    )

HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)


def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a content from a local file to base64 format."""
    # Read the audio file as binary and encode it directly to Base64
    with open(file_path, "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
    return audio_base64


with img.imports():
    # import torchaudio
    import torch
    import soundfile
    import numpy as np
    import librosa

    from loguru import logger
    from transformers.generation.streamers import BaseStreamer
    from transformers import AutoTokenizer

    from boson_multimodal.serve.serve_engine import (
        HiggsAudioServeEngine,
        HiggsAudioResponse,
        HiggsAudioStreamerDelta,
    )
    from boson_multimodal.data_types import ChatMLSample, Message, AudioContent
    from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
    from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern

    MODEL_PATH = os.getenv("LLM_MODEL", "bosonai/higgs-audio-v2-generation-3B-base")
    AUDIO_TOKENIZER_PATH = os.getenv("AUDIO_TOKENIZER_PATH", "bosonai/higgs-audio-v2-tokenizer")
    model_path = os.path.join(HF_MODEL_DIR, MODEL_PATH)
    audio_tokenizer_path = os.path.join(HF_MODEL_DIR, AUDIO_TOKENIZER_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class AsyncHiggsAudioStreamerDebug(BaseStreamer):
        def __init__(
            self,
            tokenizer: "AutoTokenizer",
            skip_prompt: bool = False,
            timeout: Optional[float] = None,
            audio_num_codebooks: int = 8,
            **decode_kwargs,
        ):
            self.tokenizer = tokenizer
            self.skip_prompt = skip_prompt
            self.timeout = timeout
            self.decode_kwargs = decode_kwargs
            self.audio_num_codebooks = audio_num_codebooks
            # Queue to store generated chunks
            self.queue = asyncio.Queue()
            self.stop_signal = None

            # Get running event loop
            self.loop = asyncio.get_running_loop()
            self.has_asyncio_timeout = hasattr(asyncio, "timeout")

            # State tracking
            self.next_tokens_are_prompt = True

        def put(self, value: torch.Tensor):
            """
            Receives tokens and processes them as either text or audio tokens.
            For text tokens, decodes and caches them until complete words are formed.
            For audio tokens, directly queues them.
            """
            # logger.info(f"{value=}, {value.shape=}, {self.next_tokens_are_prompt=}")
            if value.shape[0] > 1 and not self.next_tokens_are_prompt:
                # This is likely audio tokens (shape: [audio_num_codebooks])
                if value.shape[0] != self.audio_num_codebooks:
                    return
                delta = HiggsAudioStreamerDelta(audio_tokens=value)
                self.loop.call_soon_threadsafe(self.queue.put_nowait, delta)
                return

            # Skip prompt tokens if configured
            if self.skip_prompt and self.next_tokens_are_prompt:
                self.next_tokens_are_prompt = False
                return

            # Process as text tokens
            if len(value.shape) > 1:
                value = value[0]

            text = self.tokenizer.decode(value, **self.decode_kwargs)
            delta = HiggsAudioStreamerDelta(text=text, text_tokens=value)
            self.loop.call_soon_threadsafe(self.queue.put_nowait, delta)

        def end(self):
            """Flushes any remaining text tokens and signals the end of generation."""
            self.next_tokens_are_prompt = True
            self.loop.call_soon_threadsafe(self.queue.put_nowait, self.stop_signal)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                if self.has_asyncio_timeout:
                    async with asyncio.timeout(self.timeout):
                        value = await self.queue.get()
                else:
                    value = await asyncio.wait_for(self.queue.get(), timeout=self.timeout)
            except asyncio.TimeoutError:
                raise TimeoutError()
            else:
                if value == self.stop_signal:
                    raise StopAsyncIteration()
                else:
                    return value

    class HiggsAudioStreamServeEngine(HiggsAudioServeEngine):
        async def generate_stream(
            self,
            chat_ml_sample: ChatMLSample,
            max_new_tokens: int,
            temperature: float = 0.7,
            top_k: Optional[int] = None,
            top_p: float = 0.95,
            stop_strings: Optional[List[str]] = None,
            force_audio_gen: bool = False,
            ras_win_len: Optional[int] = 7,
            ras_win_max_num_repeat: int = 2,
            seed: Optional[int] = None,
        ):
            """
            Generate audio from a chatml sample.
            Args:
                chat_ml_sample: A chatml sample.
                max_new_tokens: The maximum number of new tokens to generate.
                temperature: The temperature to use for the generation.
                top_p: The top p to use for the generation.
                stop_strings: A list of strings to stop the generation.
                force_audio_gen: Whether to force audio generation. This ensures the model generates audio tokens rather than text tokens.
                ras_win_len: The length of the RAS window. We use 7 by default. You can disable it by setting it to None or <=0.
                ras_win_max_num_repeat: The maximum number of times to repeat the RAS window.
            Returns:
                A dictionary generator with the following keys:
                    audio: The generated audio.
                    sampling_rate: The sampling rate of the generated audio.
            """
            # Default stop strings
            if stop_strings is None:
                stop_strings = ["<|end_of_text|>", "<|eot_id|>"]
            if ras_win_len is not None and ras_win_len <= 0:
                ras_win_len = None

            with torch.no_grad():
                inputs = self._prepare_inputs(chat_ml_sample, force_audio_gen=force_audio_gen)
                print(f"{inputs=}")
                prompt_token_ids = inputs["input_ids"][0].cpu().numpy()
                print(f"{prompt_token_ids.shape} {prompt_token_ids=}")

                self._prepare_kv_caches()

                streamer = AsyncHiggsAudioStreamerDebug(self.tokenizer, skip_prompt=True)
                generation_kwargs = dict(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    stop_strings=stop_strings,
                    tokenizer=self.tokenizer,
                    do_sample=False if temperature == 0.0 else True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    past_key_values_buckets=self.kv_caches,
                    ras_win_len=ras_win_len,
                    ras_win_max_num_repeat=ras_win_max_num_repeat,
                    seed=seed,
                    streamer=streamer,
                )
                thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()

                async for delta in streamer:
                    yield delta


def print_model_params(model: torch.nn.Module, extra_info="", f=None):
    # print the number of parameters in the model
    model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(model, file=f)
    print(f"{extra_info} {model_million_params} M parameters", file=f)


@app.function(
    gpu=os.getenv("IMAGE_GPU", None),
    cpu=2.0,
    retries=0,
    image=img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_dir,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
)
async def run(func, **kwargs):
    sys.path.insert(0, "/higgs-audio/examples")
    sys.path.insert(1, "/higgs-audio/examples/serve_engine")

    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    gpu_prop = torch.cuda.get_device_properties("cuda")
    print(gpu_prop)

    if asyncio.iscoroutinefunction(func):
        await func(**kwargs)
    else:
        func(**kwargs)


def dump_model(**kwargs):
    serve_engine = HiggsAudioServeEngine(
        model_path,
        audio_tokenizer_path,
        device=device,
    )
    print(f"{serve_engine=}")
    print(f"{serve_engine.hamming_window_len=}")
    print(f"{serve_engine.kv_caches=}")
    print(f"{serve_engine.collator=}")
    print(f"{serve_engine.model.config=}")

    file_path = os.path.join(model_path, "model.txt")
    with open(file_path, "w") as f:
        print(f"text tokenizer: {serve_engine.tokenizer}", file=f)
        print_model_params(serve_engine.model, "higgs-audio-v2-generation-3B-base", f)
        print_model_params(serve_engine.audio_tokenizer, "higgs-audio-v2-tokenizer", f)


def audio_tokenize(**kwargs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_tokenizer_name_or_path = os.getenv(
        "AUDIO_TOKENIZER_PATH", "bosonai/higgs-audio-v2-tokenizer"
    )
    audio_tokenizer = load_higgs_audio_tokenizer(audio_tokenizer_name_or_path, device=device)
    print_model_params(audio_tokenizer, "higgs-audio-v2-tokenizer")
    print_model_params(audio_tokenizer.semantic_model, "hubert_base")
    # I would imagine so. A wand with a dragon heartstring core is capable of dazzling magic.
    audio_file = "/higgs-audio/examples/serve_engine/voice_examples/old_man.wav"
    vq_code = audio_tokenizer.encode(audio_file)
    print(f"{vq_code.shape} {vq_code=}")

    with torch.no_grad():
        wavform_np = audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]
    print(f"{wavform_np.shape} {wavform_np=}")
    file_path = os.path.join(ASSETS_DIR, "audio_tokenize.wav")
    soundfile.write(
        file_path,
        wavform_np,
        audio_tokenizer.sampling_rate,
        "PCM_16",
    )
    info = soundfile.info(file_path, verbose=True)
    print(info)


def audio_generate(**kwargs):
    serve_engine = HiggsAudioServeEngine(
        model_path,
        audio_tokenizer_path,
        device=device,
        torch_dtype=torch.bfloat16,
    )

    system_prompt = "Generate audio following instruction.\n\n<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>"

    zero_shot_messages = [
        Message(
            role="system",
            content=system_prompt,
        ),
        Message(
            role="user",
            content="The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years.",
            # content="太阳东升西落。这个简单的现象，人类已经观察了数千年。",
            # content="好",
        ),
    ]

    reference_text = (
        "I would imagine so. A wand with a dragon heartstring core is capable of dazzling magic."
    )
    reference_audio = encode_base64_content_from_file(
        "/higgs-audio/examples/serve_engine/voice_examples/old_man.wav"
    )
    ref_messages = [
        Message(
            role="system",
            content=system_prompt,
        ),
        Message(
            role="user",
            content=reference_text,
        ),
        Message(
            role="assistant",
            content=AudioContent(raw_audio=reference_audio, audio_url="placeholder"),
        ),
        Message(
            role="user",
            # content="Hey, everyone! Welcome back to Tech Talk Tuesdays.\n"
            # "It's your host, Alex, and today, we're diving into a topic that's become absolutely crucial in the tech world — deep learning.\n"
            # "And let's be honest, if you've been even remotely connected to tech, AI, or machine learning lately, you know that deep learning is everywhere.",
            content="嘿，各位！欢迎回到“科技对话星期二”。\n"
            "我是主持人亚历克斯，今天我们要探讨的话题在科技界已经变得至关重要——深度学习。\n"
            "说实话，如果你最近哪怕只是稍微关注一下科技、人工智能或者机器学习，你就会知道深度学习无处不在。",
        ),
    ]
    example = kwargs.get("example", "zero_shot")
    messages = ref_messages if example == "voice_clone" else zero_shot_messages

    for i in range(1):
        logger.info(f"{i} Starting generation...")
        start_time = time.time()
        output: HiggsAudioResponse = serve_engine.generate(
            chat_ml_sample=ChatMLSample(messages=messages),
            max_new_tokens=1024,
            temperature=0.3,
            top_p=0.95,
            top_k=50,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"],
        )
        elapsed_time = time.time() - start_time

        logger.info(f"{i}. Generated text:\n{output.generated_text}")
        gen_audio_path = os.path.join(ASSETS_DIR, f"higgsv2_gen_audio_{i}.wav")

        soundfile.write(gen_audio_path, output.audio, output.sampling_rate)
        info = soundfile.info(gen_audio_path, verbose=True)

        # torchaudio.save(
        #    gen_audio_path, torch.from_numpy(output.audio)[None, :], output.sampling_rate
        # )

        print(info)
        logger.info(
            f"{i}. Generation time: {elapsed_time:.2f} seconds, duration: {info.duration:.2f} seconds, RTF: {(elapsed_time / info.duration):.2f}"
        )
        logger.info(f"Saved audio to {gen_audio_path}")


def run_hf_example(**kwargs):
    from run_hf_example import test

    example = kwargs.get("example", "zero_shot")
    gen_audio_path = os.path.join(ASSETS_DIR, f"higgsv2_gen_{example}.wav")
    test(example, model_path, audio_tokenizer_path, gen_audio_path)


def generation(**kwargs):
    from generation import gen

    kwargs.pop("example")
    kwargs["model_path"] = model_path
    kwargs["audio_tokenizer"] = audio_tokenizer_path
    kwargs["transcript"] = os.path.join("/higgs-audio/examples", kwargs["transcript"])
    kwargs["scene_prompt"] = os.path.join("/higgs-audio/examples", kwargs["scene_prompt"])
    kwargs["out_path"] = os.path.join(ASSETS_DIR, kwargs.get("out_path", "generation.wav"))
    print(f"{kwargs=}")
    gen(**kwargs)

    info = soundfile.info(kwargs["out_path"], verbose=True)
    print(info)


async def audio_generate_stream(**kwargs):
    # serve_engine = HiggsAudioStreamServeEngine(
    serve_engine = HiggsAudioServeEngine(
        model_path,
        audio_tokenizer_path,
        device=device,
        torch_dtype=torch.bfloat16,
    )

    system_prompt = "Generate audio following instruction.\n\n<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>"

    messages = [
        Message(
            role="system",
            content=system_prompt,
        ),
        Message(
            role="user",
            # content="The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years.",
            content="太阳东升西落。这个简单的现象，人类已经观察了数千年。",
            # content="好",
            # content="好！",
            # content="good",
        ),
    ]
    for i in range(1):
        logger.info(f"{i} Starting generation...")
        # output = serve_engine.generate_stream(
        output = serve_engine.generate_delta_stream(
            chat_ml_sample=ChatMLSample(messages=messages),
            max_new_tokens=1024,
            temperature=0.3,
            top_p=0.95,
            top_k=50,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"],
        )

        audio_tokens = []
        audio_tensor = None
        CHUNK_SIZE = 16
        seq_len = 0
        waveform_list = []

        chunk_overlap_duration = kwargs.get("chunk_overlap_duration", 0.04)
        cross_fade_samples = int(
            chunk_overlap_duration * serve_engine.audio_tokenizer.sampling_rate
        )
        fade_out = np.linspace(1, 0, cross_fade_samples)
        fade_in = np.linspace(0, 1, cross_fade_samples)

        with torch.inference_mode():
            async for delta in output:
                if delta.text:
                    print(delta.text, end="", flush=True)

                if delta.audio_tokens is None:
                    continue

                if torch.all(delta.audio_tokens == 1025):
                    break

                # print(f"{delta.audio_tokens=}")
                audio_tokens.append(delta.audio_tokens[:, None])
                audio_tensor = torch.cat(audio_tokens, dim=-1)
                print(f"{audio_tensor=}")

                if torch.all(delta.audio_tokens != 1024):
                    seq_len += 1
                    print(f"{delta.audio_tokens=} {seq_len=}")
                if seq_len > 0 and seq_len % CHUNK_SIZE == 0:
                    vq_code = (
                        revert_delay_pattern(audio_tensor, start_idx=seq_len - CHUNK_SIZE + 1)
                        .clip(0, 1023)
                        .to(device)
                    )

                    print(f"vq_code shape: {vq_code.shape} {vq_code=}")
                    waveform_numpy = serve_engine.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]
                    waveform_list.append(waveform_numpy)

                    # gen_audio_path = os.path.join(ASSETS_DIR, f"higgsv2_gen_audio_{i}_{seq_len}.wav")
                    # soundfile.write(gen_audio_path, waveform_numpy, serve_engine.audio_tokenizer.sampling_rate)
                    # info = soundfile.info(gen_audio_path, verbose=True)
                    # print(f"\nSaved audio chunk to {gen_audio_path}, duration: {info.duration:.2f} seconds")

            if seq_len > 0 and seq_len % CHUNK_SIZE != 0 and audio_tensor is not None:
                print(f"{audio_tensor=} {seq_len=}")
                vq_code = (
                    revert_delay_pattern(audio_tensor, start_idx=seq_len - seq_len % CHUNK_SIZE + 1)
                    .clip(0, 1023)
                    .to(device)
                )

                print(f"vq_code shape: {vq_code.shape} {vq_code=}")
                waveform_numpy = serve_engine.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]
                waveform_list.append(waveform_numpy)

            # new_audio = np.concatenate(waveform_list)

            for j, audio in enumerate(waveform_list):
                if j == 0:
                    new_audio = audio[:-cross_fade_samples]
                else:
                    cross_faded_overlap = (
                        waveform_list[j - 1][-cross_fade_samples:] * fade_out
                        + audio[:cross_fade_samples] * fade_in
                    )
                    new_audio = np.concatenate(
                        [
                            new_audio,
                            cross_faded_overlap,
                            audio[cross_fade_samples:-cross_fade_samples],
                        ]
                    )
            new_audio = np.concatenate([new_audio, audio[-cross_fade_samples:]])
            new_audio, _ = librosa.effects.trim(new_audio, top_db=60)

            gen_audio_path = os.path.join(ASSETS_DIR, f"higgsv2_gen_audio_stream_{i}.wav")
            soundfile.write(
                gen_audio_path,
                new_audio,
                serve_engine.audio_tokenizer.sampling_rate,
                "PCM_16",
            )
            info = soundfile.info(gen_audio_path, verbose=True)
            print(info)
            print(f"\nSaved audio chunk to {gen_audio_path}, duration: {info.duration:.2f} seconds")


async def achatbot_audio_generate_stream(**kwargs):
    from achatbot.common.interface import ITts
    from achatbot.modules.speech.tts import TTSEnvInit
    from achatbot.common.session import Session
    from achatbot.common.types import SessionCtx
    from achatbot.common.logger import Logger

    Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)

    tts: ITts = TTSEnvInit.initTTSEngine("tts_higgs")
    session = Session(**SessionCtx("test_tts_client_id").__dict__)
    voices = tts.get_voices()
    assert len(voices) == 1
    print(voices)
    tts.set_voice(
        "/higgs-audio/examples/serve_engine/voice_examples/old_man.wav",
        ref_speaker="test_speaker",
        ref_text="I would imagine so. A wand with a dragon heartstring core is capable of dazzling magic.",
    )
    add_voices = tts.get_voices()
    assert len(add_voices) == len(voices) + 1
    print(add_voices)

    stream_info = tts.get_stream_info()
    print(f"stream_info:{stream_info}")

    session.ctx.state["tts_text"] = "太阳东升西落。这个简单的现象，人类已经观察了数千年。"
    session.ctx.state["temperature"] = 0.3
    session.ctx.state["chunk_size"] = 8
    # session.ctx.state["ref_speaker"] = "test_speaker"
    # print(session.ctx)

    times = []
    start_time = time.perf_counter()
    waveform_list = []
    # for chunk_bytes in tts.synthesize_sync(session):
    async for chunk_bytes in tts.synthesize(session):
        times.append(time.perf_counter() - start_time)
        waveform_numpy = np.frombuffer(chunk_bytes, dtype=stream_info["np_dtype"])
        waveform_list.append(waveform_numpy)
        start_time = time.perf_counter()
    print(f"generate first time: {times[0]} s, cost time: {sum(times)} s")

    chunk_overlap_duration = kwargs.get("chunk_overlap_duration", 0.04)
    cross_fade_samples = int(chunk_overlap_duration * stream_info["rate"])
    fade_out = np.linspace(1, 0, cross_fade_samples)
    fade_in = np.linspace(0, 1, cross_fade_samples)

    file_name = f"test_{tts.TAG}.wav"
    file_path = os.path.join(ASSETS_DIR, file_name)

    # new_audio = np.concatenate(waveform_list)

    for j, audio in enumerate(waveform_list):
        if j == 0:
            new_audio = audio[:-cross_fade_samples]
        else:
            cross_faded_overlap = (
                waveform_list[j - 1][-cross_fade_samples:] * fade_out
                + audio[:cross_fade_samples] * fade_in
            )
            new_audio = np.concatenate(
                [
                    new_audio,
                    cross_faded_overlap,
                    audio[cross_fade_samples:-cross_fade_samples],
                ]
            )
    new_audio = np.concatenate([new_audio, audio[-cross_fade_samples:]])
    # 去除静音部分
    new_audio, _ = librosa.effects.trim(new_audio, top_db=60)
    soundfile.write(
        file_path,
        new_audio,
        stream_info["rate"],
        "PCM_16",
    )
    info = soundfile.info(file_path, verbose=True)
    print(info)
    print(f"\nSaved audio chunk to {file_path}, duration: {info.duration:.2f} seconds")


"""
    Higgs-Audio is an end-to-end multimodal model with the capability to understand and generate text / audio.

    Consider the following example for mixed text/audio understanding / generation:

    - input_tokens: <text_token1><|audio_bos|>[AUDIO]<|audio_eos|><text_token2><|audio_bos|>[AUDIO]<|audio_eos|><text_token4>
    - input_tokens: <text_token1><|audio_bos|>[AUDIO]<|audio_eos|><text_token2><|audio_out_bos|>[AUDIO_OUT]<|audio_eos|><text_token4>

    We will fill [AUDIO] with the audio features extracted by Whisper and fill [AUDIO_OUT] with the audio tokens. when use audio understand

    Consider the following example for mixed text/audio generation:

    text: <|audio_out_bos|>    MASK           MASK           MASK          MASK               MASK         <|audio_eos|> [text_token1]
    audio:     MASK    <|audio_stream_bos|> [audio_token1] [audio_token2] [audio_token3] <|audio_stream_eos|>   MASK           MASK
    token_type: 0               1              1              1             1                  1                 0              0
"""
"""
NOTE:  if use do_sample is True, see _sample code
HiggsAudioModel
    # Built on top of GenerationMixin._sample.
    # We revise the implementation to support generating both audio / text.
    def _sample()
"""
"""
TIP:
- 中文简单的词，比如 "好", 有时没有生成声音（人类所能分辨的音频段)
"""

"""
IMAGE_GPU=L4 modal run src/llm/transformers/higgs_audio.py --task dump_model

IMAGE_GPU=L4 modal run src/llm/transformers/higgs_audio.py --task audio_generate
IMAGE_GPU=L4 modal run src/llm/transformers/higgs_audio.py --task audio_generate --example voice_clone

IMAGE_GPU=L4 modal run src/llm/transformers/higgs_audio.py --task audio_generate_stream

APP_NAME=achatbot IMAGE_GPU=L4 modal run src/llm/transformers/higgs_audio.py --task achatbot_audio_generate_stream 

IMAGE_GPU=L4 modal run src/llm/transformers/higgs_audio.py --task run_hf_example --example zero_shot
IMAGE_GPU=L4 modal run src/llm/transformers/higgs_audio.py --task run_hf_example --example voice_clone 
IMAGE_GPU=L4 modal run src/llm/transformers/higgs_audio.py --task run_hf_example --example interleaved_dialogue

# https://github.com/boson-ai/higgs-audio/blob/main/examples/README.md

# ref clone voice
IMAGE_GPU=L4 modal run src/llm/transformers/higgs_audio.py --task generation \
    --transcript transcript/single_speaker/en_dl.txt \
    --ref-audio broom_salesman \
    --seed 12345 \
    --out-path ref_broom_salesman.wav
IMAGE_GPU=L4 modal run src/llm/transformers/higgs_audio.py --task generation \
    --transcript transcript/single_speaker/en_dl.txt \
    --ref-audio belinda \
    --seed 12345 \
    --out-path ref_belinda.wav
IMAGE_GPU=L4 modal run src/llm/transformers/higgs_audio.py --task generation \
    --transcript transcript/single_speaker/en_dl.txt \
    --scene-prompt empty \
    --ref-audio zh_man_sichuan \
    --temperature 0.3 \
    --seed 12345 \
    --out-path ref_zh_man_sichuan.wav

# Random voice
IMAGE_GPU=L4 modal run src/llm/transformers/higgs_audio.py --task generation \
    --transcript transcript/single_speaker/en_dl.txt \
    --seed 12345 \
    --out-path random_en.wav
IMAGE_GPU=L4 modal run src/llm/transformers/higgs_audio.py --task generation \
    --transcript transcript/single_speaker/zh_ai.txt \
    --seed 12345 \
    --out-path random_zh.wav

# Describe speaker characteristics with text
## Male British Accent
IMAGE_GPU=L4 modal run src/llm/transformers/higgs_audio.py --task generation \
    --transcript transcript/single_speaker/en_dl.txt \
    --ref-audio profile:male_en_british \
    --seed 12345 \
    --out-path character_male_en_british.wav
## Female British Accent
IMAGE_GPU=L4 modal run src/llm/transformers/higgs_audio.py --task generation \
    --transcript transcript/single_speaker/en_dl.txt \
    --ref-audio profile:female_en_british \
    --seed 12345 \
    --out-path character_female_en_british.wav

# Chunking for long-form audio generation
IMAGE_GPU=L4 modal run src/llm/transformers/higgs_audio.py --task generation \
    --scene-prompt scene_prompts/reading_blog.txt \
    --transcript transcript/single_speaker/en_higgs_audio_blog.md \
    --ref-audio en_man \
    --chunk-method word \
    --temperature 0.3 \
    --generation-chunk-buffer-size 2 \
    --seed 12345 \
    --out-path long_chunk_buffer_audio_blog.wav

# Experimental and Emergent Capabilities
## Hum a tune with the cloned voice
IMAGE_GPU=L4 modal run src/llm/transformers/higgs_audio.py --task generation \
    --transcript transcript/single_speaker/experimental/en_humming.txt \
    --ref-audio en_woman \
    --temperature 1.0 \
    --ras-win-len 0 \
    --seed 12345 \
    --out-path en_woman_tune.wav
## Read the sentence while adding background music (BGM)
IMAGE_GPU=L4 modal run src/llm/transformers/higgs_audio.py --task generation \
    --transcript transcript/single_speaker/experimental/en_bgm.txt \
    --ref-audio en_woman \
    --temperature 1.0 \
    --ras-win-len 0 \
    --ref-audio-in-system-message \
    --seed 123456 \
    --out-path en_woman_bgm.wav

# Multi-speaker Audio Generation
## Multi-voice random
IMAGE_GPU=L4 modal run src/llm/transformers/higgs_audio.py --task generation \
    --transcript transcript/multi_speaker/en_argument.txt \
    --seed 12345 \
    --out-path multi_speaker_random.wav

## Multi-voice clone
IMAGE_GPU=L4 modal run src/llm/transformers/higgs_audio.py --task generation \
    --transcript transcript/multi_speaker/en_argument.txt \
    --ref-audio belinda,broom_salesman \
    --ref-audio-in-system-message \
    --chunk-method speaker \
    --seed 12345 \
    --out-path multi_speaker_belinda,broom_salesman.wav
IMAGE_GPU=L4 modal run src/llm/transformers/higgs_audio.py --task generation \
    --transcript transcript/multi_speaker/en_higgs.txt \
    --ref-audio broom_salesman,belinda\
    --ref-audio-in-system-message \
    --chunk-method speaker \
    --chunk-max-num-turns 2 \
    --seed 12345 \
    --out-path multi_speaker_broom_salesman,belinda.wav

"""


@app.local_entrypoint()
def main(
    task: str = "dump_model",
    example: str = "zero_shot",
    *,
    max_new_tokens: int = 2048,
    transcript: str = "transcript/single_speaker/en_dl.txt",
    scene_prompt: str = "scene_prompts/quiet_indoor.txt",
    temperature: float = 0.7,
    top_k: int = 20,
    top_p: float = 0.95,
    ras_win_len: int = 7,
    ras_win_max_num_repeat: int = 2,
    ref_audio: str = None,
    ref_audio_in_system_message: bool = False,
    chunk_method: str = None,  # None, "speaker", "word"
    chunk_max_word_num: int = 200,
    chunk_max_num_turns: int = 1,
    generation_chunk_buffer_size: int = None,
    seed: int = 42,
    device_id: int = None,
    use_static_kv_cache: int = 1,
    out_path: str = "generation.wav",
):
    print(task)
    tasks = {
        "dump_model": dump_model,
        "audio_tokenize": audio_tokenize,
        "audio_generate": audio_generate,
        "audio_generate_stream": audio_generate_stream,
        "run_hf_example": run_hf_example,
        "generation": generation,
        "achatbot_audio_generate_stream": achatbot_audio_generate_stream,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(
        tasks[task],
        example=example,
        max_new_tokens=max_new_tokens,
        transcript=transcript,
        scene_prompt=scene_prompt,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        ras_win_len=ras_win_len,
        ras_win_max_num_repeat=ras_win_max_num_repeat,
        ref_audio=ref_audio,
        ref_audio_in_system_message=ref_audio_in_system_message,
        chunk_method=chunk_method,
        chunk_max_word_num=chunk_max_word_num,
        chunk_max_num_turns=chunk_max_num_turns,
        generation_chunk_buffer_size=generation_chunk_buffer_size,
        seed=seed,
        device_id=device_id,
        use_static_kv_cache=use_static_kv_cache,
        out_path=out_path,
    )
