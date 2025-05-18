import math
import os
import sys
from random import random
import re
from threading import Thread
from time import perf_counter
import time
from typing import Optional
import uuid

import modal


app = modal.App("vita_audio")
vita_audio_img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "clang", "cmake", "ninja-build")
    .run_commands(
        "git clone -b feat/achatbot https://github.com/weedge/VITA-Audio.git",
        "cd /VITA-Audio && git submodule update --init --recursive",
        "cd /VITA-Audio && pip install -q -r requirements_ds_gpu.txt",
    )
    .pip_install(
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "torchvision==0.21.0",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install("wheel")
    # https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    .run_commands(
        "cd /VITA-Audio && git pull origin feat/achatbot",
        "cd /VITA-Audio && git checkout 32fb8180fcd7a4bbaa8c9dd77928c78cb7fd26ce",
    )
    .run_commands(
        "git clone https://github.com/weedge/CosyVoice.git",
        "cd /CosyVoice && git checkout 6507762ba72ecfd4f9c15e17f543c8aaeef2fd8f",
        "pip install rich",
    )
    .pip_install("snac")
    .run_commands(
        "git clone https://github.com/weedge/Spark-TTS.git",
        "cd /Spark-TTS && git checkout 244144322c738a64d98ce51247d8de75a74e9449",
    )
    .pip_install("einops", "einx", "omegaconf", "packaging", "safetensors", "soxr")
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TQDM_DISABLE": "1",
            "AUDIO_TOKENIZER_TYPE": os.getenv("AUDIO_TOKENIZER_TYPE", "sensevoice_glm4voice"),
            # VITA-MLLM/VITA-Audio-Boost | VITA-MLLM/VITA-Audio-Balance
            "MTP_LLM_MODEL": os.getenv("MTP_LLM_MODEL", "VITA-MLLM/VITA-Audio-Balance"),
        }
    )
)

HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)

with vita_audio_img.imports():
    import subprocess
    import torch
    import numpy as np
    import tqdm
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    from transformers.generation import GenerationConfig

    sys.path.append("/VITA-Audio")
    sys.path.append("/VITA-Audio/third_party/GLM-4-Voice/")
    sys.path.append("/VITA-Audio/third_party/GLM-4-Voice/cosyvoice/")
    sys.path.append("/VITA-Audio/third_party/GLM-4-Voice/third_party/Matcha-TTS/")

    from vita_audio.tokenizer import get_audio_tokenizer
    from vita_audio.data.processor.audio_processor import add_audio_input_contiguous

    # audio vq codec model | cosyvoice (glm-4-voice-tokenizer or FunAudioLLM/CosyVoice2-0.5B tokenizer + vocoder)
    audio_tokenizer_model_path = None
    sense_voice_model_path = None  # sensevoice_small model
    flow_path = None
    model_path = None

    # glm4voice | cosyvoice2 | snac24khz | sensevoice_sparktts | sensevoice_glm4voice
    support_types = [
        "glm4voice",
        "sensevoice_glm4voice",
        # "snac24khz",
        # "cosyvoice2",
        # "sensevoice_sparktts",
    ]
    audio_tokenizer_type = os.getenv("AUDIO_TOKENIZER_TYPE", "sensevoice_glm4voice")
    print(f"{audio_tokenizer_type=}")
    mtp_llm_model = os.path.join(
        HF_MODEL_DIR, os.getenv("MTP_LLM_MODEL", "VITA-MLLM/VITA-Audio-Balance")
    )
    print(f"{mtp_llm_model=}")

    if audio_tokenizer_type == "sensevoice_glm4voice":
        # sensevoice_glm4voice
        # (sensevoice_small WaveFrontend extract audio feature)
        sense_voice_model_path = os.path.join(HF_MODEL_DIR, "FunAudioLLM/SenseVoiceSmall")
        # glm-4-voice-decoder (flow + hift) from cosyvoice (support streaming)
        flow_path = os.path.join(HF_MODEL_DIR, "THUDM/glm-4-voice-decoder")
        # sensevoice_small encoder(no ctc head) + qwen2 model (no mtp)
        # model_path = os.path.join(HF_MODEL_DIR, "VITA-MLLM/VITA-Audio-Plus-Vanilla")
        model_path = mtp_llm_model

    if audio_tokenizer_type == "glm4voice":
        audio_tokenizer_model_path = os.path.join(HF_MODEL_DIR, "THUDM/glm-4-voice-tokenizer")
        flow_path = os.path.join(HF_MODEL_DIR, "THUDM/glm-4-voice-decoder")

        model_path = mtp_llm_model
        # run is ok, but don't use this model
        # model_path = os.path.join(HF_MODEL_DIR, "VITA-MLLM/VITA-Audio-Plus-Vanilla")

    if audio_tokenizer_type == "cosyvoice2":
        sys.path.insert(0, "/CosyVoice")
        # (sensevoice_small WaveFrontend extract audio feature)
        audio_tokenizer_model_path = os.path.join(HF_MODEL_DIR, "FunAudioLLM/CosyVoice2-0.5B")

        model_path = mtp_llm_model

        # run is ok, but don't use this model
        # model_path = os.path.join(HF_MODEL_DIR, "VITA-MLLM/VITA-Audio-Plus-Vanilla")

    if audio_tokenizer_type == "snac24khz":
        audio_tokenizer_model_path = os.path.join(HF_MODEL_DIR, "hubertsiuzdak/snac_24khz")

        model_path = mtp_llm_model

        # run is ok, but don't use this model
        # model_path = os.path.join(HF_MODEL_DIR, "VITA-MLLM/VITA-Audio-Plus-Vanilla")

    if audio_tokenizer_type == "sensevoice_sparktts":
        sys.path.insert(0, "/Spark-TTS")
        # bicodec (vq codec)
        audio_tokenizer_model_path = os.path.join(HF_MODEL_DIR, "SparkAudio/Spark-TTS-0.5B")

        model_path = mtp_llm_model

        # run is ok, but don't use this model
        # sense_voice_model_path = os.path.join(HF_MODEL_DIR, "FunAudioLLM/SenseVoiceSmall")
        # model_path = os.path.join(HF_MODEL_DIR, "VITA-MLLM/VITA-Audio-Plus-Vanilla")


@app.function(
    gpu=os.getenv("IMAGE_GPU", None),
    cpu=2.0,
    retries=1,
    image=vita_audio_img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_dir,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
)
def run(func):
    if audio_tokenizer_type not in support_types:
        print(
            f"Invalid audio_tokenizer_type: {audio_tokenizer_type}, please choose one of the following: {support_types}"
        )
        return
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    gpu_prop = torch.cuda.get_device_properties("cuda")
    print(gpu_prop)

    func()


def tokenize():
    from evaluation.get_chat_template import qwen2_chat_template as chat_template

    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    print(f"{config=}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        chat_template=chat_template,
    )
    # print(f"{tokenizer=}")
    print(f"{tokenizer.get_chat_template()=}")


def dump_model():
    # https://huggingface.co/VITA-MLLM/VITA-Audio-Boost/blob/main/modeling_qwen2.py#L781
    # https://huggingface.co/VITA-MLLM/VITA-Audio-Balance/blob/main/modeling_qwen2.py#L781
    # https://huggingface.co/VITA-MLLM/VITA-Audio-Plus-Vanilla/blob/main/modeling_qwen2.py#L834
    for name, model_path in {
        "Boost": os.path.join(HF_MODEL_DIR, "VITA-MLLM/VITA-Audio-Boost"),
        "Balance": os.path.join(HF_MODEL_DIR, "VITA-MLLM/VITA-Audio-Balance"),
        "Vanilla": os.path.join(HF_MODEL_DIR, "VITA-MLLM/VITA-Audio-Plus-Vanilla"),
    }.items():
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).eval()
        print_model_params(model, f"Qwen2MTP_AR_LLM_{name}")
        del model
        torch.cuda.empty_cache()

    audio_tokenizer = get_audio_tokenizer(
        model_type=audio_tokenizer_type,
        flow_path=flow_path,
        rank=0,
        audio_tokenizer_model_path=audio_tokenizer_model_path,
        sense_voice_model_path=sense_voice_model_path,
    )
    print(f"{audio_tokenizer=}")

    audio_tokenizer.load_model()
    print(f"SenseVoiceSmall args: {audio_tokenizer.kwargs}")
    print_model_params(audio_tokenizer.sensevoice_model, "audio_tokenizer.audio_encoder.sensevoice")
    print_model_params(audio_tokenizer.whisper_model, "audio_tokenizer.audio_encoder.whisper")

    print_model_params(audio_tokenizer.audio_decoder.flow, "audio_tokenizer.audio_decoder.flow")
    print_model_params(audio_tokenizer.audio_decoder.hift, "audio_tokenizer.audio_decoder.hift")


def benchmark_llm():
    s2s_inference = S2SInference(
        model_path,
        audio_tokenizer_type,
        audio_tokenizer_model_path=audio_tokenizer_model_path,
        sense_voice_model_path=sense_voice_model_path,
        flow_path=flow_path,
    )
    for mtp_inference_mode, tag in zip(
        [
            [8192, 0],
            [1, 4, 3, 8, 4, 10],
            [1, 10, 4, 10],
            [1, 10],
        ],
        [
            "Vanilla",
            "Balance",
            "Boost",
            "Turbo",
        ],
    ):
        print("=" * 100)
        print("benchmark_llm")
        print(f"{tag}")

        s2s_inference.benchmark_forward(mtp_inference_mode)

        s2s_inference.benchmark_generate(mtp_inference_mode)

        generated_text = ""
        for new_text in s2s_inference.benchmark_generate_stream(
            mtp_inference_mode=mtp_inference_mode
        ):
            generated_text += new_text
            # print(new_text, end="", flush=True)


def benchmark_sts():
    import torchaudio

    s2s_inference = S2SInference(
        model_path,
        audio_tokenizer_type,
        audio_tokenizer_model_path=audio_tokenizer_model_path,
        sense_voice_model_path=sense_voice_model_path,
        flow_path=flow_path,
    )

    audio_paths = [
        "/VITA-Audio/asset/介绍一下上海.wav",
        "/VITA-Audio/asset/发表一个悲伤的演讲.wav",
        "/VITA-Audio/asset/发表一个振奋人心的演讲.wav",
    ]
    output_dir = ASSETS_DIR

    for _ in range(10):
        print("=" * 100)
        print("benchmark_sts")
        audio_path = random.choice(audio_paths)
        print(f"{audio_path}")

        start = time.time()
        audio_idx = 0
        generated_text = ""
        all_tts_speech = []
        past_tts_speech_len = 0
        for new_text in s2s_inference.run_infer_stream(audio_path=audio_path):
            # print(new_text, end="", flush=True)

            generated_text += new_text

            if new_text == "<|end_of_audio|>":
                audio_tokens = extract_token_ids_as_int(generated_text)

                tts_speech = s2s_inference.audio_tokenizer.decode(audio_tokens, option_steps=1)
                tts_speech = tts_speech[past_tts_speech_len:]
                past_tts_speech_len += len(tts_speech)
                all_tts_speech.append(tts_speech)

                end = time.time()
                if audio_idx == 0:
                    print(audio_tokens)
                print(f"{audio_idx} audio chunk {end - start}")

                wav_path = os.path.join(output_dir, audio_path[:-4] + f"_{audio_idx}.wav")
                os.makedirs(os.path.dirname(wav_path), exist_ok=True)
                torchaudio.save(wav_path, tts_speech.unsqueeze(0), 22050, format="wav")

                audio_idx += 1
                start = time.time()

        wav_path = os.path.join(output_dir, audio_path[:-4] + ".wav")
        tts_speech = torch.cat(all_tts_speech, dim=0)
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        torchaudio.save(wav_path, tts_speech.unsqueeze(0), 22050, format="wav")


# ==============================================================
# Text
def text():
    s2s_inference = S2SInference(
        model_path,
        audio_tokenizer_type,
        sense_voice_model_path=None,
        flow_path=None,
    )

    for text in [
        # "How many helicopters can a human eat in one sitting?",
        # "你叫什么名字？",
        # "写一首诗",
        "介绍一下上海",
    ]:
        print("=" * 100)
        print("text_task")
        print(f"{text=}")

        start = perf_counter()
        output, _ = s2s_inference.run_infer(
            message=text,
            mode=None,
            # do_sample=True,
            mtp_inference_mode=[8192, 0],
        )
        print(f"{output=}", flush=True)
        print(f"cost: {perf_counter()-start} s")


def text_audio():
    import torchaudio

    s2s_inference = S2SInference(
        model_path,
        audio_tokenizer_type,
        sense_voice_model_path=None,
        flow_path=flow_path,
    )

    for text in [
        "How many helicopters can a human eat in one sitting?",
        "你叫什么名字？",
        "写一首诗",
        "介绍一下北京",
    ]:
        print("=" * 100)
        print("text_audio_task")
        print(f"{text=}")

        start = perf_counter()
        output, tts_speech = s2s_inference.run_infer(
            message=text,
            mode="luke",
            do_sample=False,
            # mtp_inference_mode=[8192, 0],
        )
        print(f"{output=}", flush=True)
        print(f"{tts_speech.shape=}", flush=True)
        print(f"cost: {perf_counter()-start} s")

        file_name = text[:16]
        wav_path = os.path.join(ASSETS_DIR, f"sts_{file_name}.wav")
        print(f"save to {wav_path=}")
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        torchaudio.save(wav_path, tts_speech.unsqueeze(0), 22050, format="wav")


# ==============================================================
# Text stream
def text_stream():
    s2s_inference = S2SInference(
        model_path,
        audio_tokenizer_type,
        sense_voice_model_path=None,
        flow_path=None,
    )
    for text in [
        "你叫什么名字？",
        "讲一个儿童故事",
    ]:
        print("=" * 100)
        print("text_stream_task")
        print(f"{text=}")

        times = []
        start_time = perf_counter()
        generated_text = ""
        for new_text in s2s_inference.run_infer_stream(
            message=text,
            mode=None,
            # do_sample=True,
            mtp_inference_mode=[8192, 0],
        ):
            times.append(perf_counter() - start_time)
            generated_text += new_text
            print(new_text, end="", flush=True)
            start_time = perf_counter()

        print(
            f"\ngenerate [{generated_text}] first token cost time: {times[0]} s, {len(times)} tokens cost time: {sum(times)} s\n"
        )


def text_audio_stream():
    import soundfile as sf

    s2s_inference = S2SInference(
        model_path,
        audio_tokenizer_type,
        sense_voice_model_path=None,
        flow_path=flow_path,
    )
    for text in [
        "你叫什么名字？",
        # "请讲一个儿童故事。",
    ]:
        print("=" * 100)
        print("text_audio_stream_task")
        print(f"{text=}")

        times = []
        start_time = perf_counter()
        generated_text = ""
        for new_text in s2s_inference.run_infer_stream(
            message=text,
            mode="luke",
            do_sample=True,
            # mtp_inference_mode=[8192, 0],
        ):
            times.append(perf_counter() - start_time)
            generated_text += new_text
            print(new_text, end=" ", flush=True)
            start_time = perf_counter()

        print(
            f"\ngenerate [{generated_text}] first token cost time: {times[0]} s, {len(times)} tokens cost time: {sum(times)} s\n"
        )
        file_name = text[:16]
        audio_decode_time = []
        audios = []
        audio_segments = find_audio_segments_regex(generated_text)
        for audio_idx, audio_segment in enumerate(audio_segments):
            start = perf_counter()

            audio_tokens = extract_token_ids_as_int(audio_segment)
            print(f"{audio_tokens=}")

            tts_speech = s2s_inference.audio_tokenizer.decode(audio_tokens)
            audio_decode_time.append(perf_counter() - start)

            # wav_path = os.path.join(ASSETS_DIR, f"qa_stream_{audio_idx}_{file_name}.wav")
            # print(f"save to {wav_path=}")
            # torchaudio.save(wav_path, tts_speech.unsqueeze(0), 22050, format="wav")

            audios.append(tts_speech.cpu().numpy())

        wav_path = os.path.join(ASSETS_DIR, f"sts_stream_{file_name}.wav")
        print(f"save to {wav_path=}")
        sf.write(wav_path, np.concatenate(audios), samplerate=22050)
        info = sf.info(wav_path, verbose=True)
        print(
            f"\ngenerate first audio segment cost time: {audio_decode_time[0]} s, {len(audio_decode_time)} segment cost time: {sum(audio_decode_time)} s | wav duration: {info.duration} s | audio tokenizer decode RTF: {sum(audio_decode_time)/info.duration} \n"
        )


# ==============================================================
# S2S
def sts():
    import torchaudio

    output_dir = ASSETS_DIR
    s2s_inference = S2SInference(
        model_path,
        audio_tokenizer_type,
        audio_tokenizer_model_path=audio_tokenizer_model_path,
        sense_voice_model_path=sense_voice_model_path,
        flow_path=flow_path,
    )

    for audio_path in [
        "/VITA-Audio/asset/介绍一下上海.wav",
        # "/VITA-Audio/asset/发表一个悲伤的演讲.wav",
        # "/VITA-Audio/asset/发表一个振奋人心的演讲.wav",
        # "/VITA-Audio/asset/piano.mp3",
    ]:
        print("=" * 100)
        print("sts_task")
        print(f"{audio_path=}")
        subprocess.run(f"cp {audio_path} {ASSETS_DIR}", shell=True)

        start_time = perf_counter()
        output, tts_speech = s2s_inference.run_infer(
            audio_path=audio_path,
        )
        print(f"{output=}")
        print(f"{tts_speech.shape=}", flush=True)
        print(f"cost time: {perf_counter()-start_time:.6f} s")

        file_name = os.path.basename(audio_path)
        wav_path = os.path.join(output_dir, f"sts_{file_name}")
        print(f"save to {wav_path=}")
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        torchaudio.save(wav_path, tts_speech.unsqueeze(0), 22050, format="wav")


# ==============================================================
# S2S stream
def sts_stream():
    """
    audio decode need look-ahead
    """
    import torchaudio
    import soundfile as sf

    s2s_inference = S2SInference(
        model_path,
        audio_tokenizer_type,
        audio_tokenizer_model_path=audio_tokenizer_model_path,
        sense_voice_model_path=sense_voice_model_path,
        flow_path=flow_path,
    )

    for audio_path in [
        "/VITA-Audio/asset/介绍一下上海.wav",  # warmup
        "/VITA-Audio/asset/介绍一下上海.wav",
    ]:
        print("=" * 100)
        print("sts_stream_task")
        print(f"{audio_path=}")

        times = []
        start_time = perf_counter()
        generated_text = ""
        for new_text in s2s_inference.run_infer_stream(audio_path=audio_path):
            times.append(perf_counter() - start_time)
            generated_text += new_text
            print(new_text, end="")
            start_time = perf_counter()
        print(
            f"\ngenerate [{generated_text}] first token cost time: {times[0]} s, {len(times)} tokens cost time: {sum(times)} s\n"
        )

        file_name = os.path.basename(audio_path)
        audio_decode_time = []
        audios = []
        audio_segments = find_audio_segments_regex(generated_text)
        for audio_idx, audio_segment in enumerate(audio_segments):
            start = perf_counter()

            audio_tokens = extract_token_ids_as_int(audio_segment)
            print(f"{audio_tokens=}")

            tts_speech = s2s_inference.audio_tokenizer.decode(audio_tokens)
            audio_decode_time.append(perf_counter() - start)

            # wav_path = os.path.join(ASSETS_DIR, f"qa_stream_{audio_idx}_{file_name}")
            # print(f"save to {wav_path=}")
            # torchaudio.save(wav_path, tts_speech.unsqueeze(0), 22050, format="wav")

            audios.append(tts_speech.cpu().numpy())

        wav_path = os.path.join(ASSETS_DIR, f"sts_stream_{file_name}")
        print(f"save to {wav_path=}")
        sf.write(wav_path, np.concatenate(audios), samplerate=22050)
        info = sf.info(wav_path, verbose=True)

        print(
            f"\ngenerate first audio segment cost time: {audio_decode_time[0]} s, {len(audio_decode_time)} segment cost time: {sum(audio_decode_time)} s | wav duration: {info.duration} s | audio tokenizer decode RTF: {sum(audio_decode_time)/info.duration} \n"
        )


# ==============================================================
# ASR
def asr():
    s2s_inference = S2SInference(
        model_path,
        audio_tokenizer_type,
        audio_tokenizer_model_path=audio_tokenizer_model_path,
        sense_voice_model_path=sense_voice_model_path,
        flow_path=None,
    )

    for audio_path in [
        "/VITA-Audio/asset/介绍一下上海.wav",
        "/VITA-Audio/asset/发表一个悲伤的演讲.wav",
        "/VITA-Audio/asset/发表一个振奋人心的演讲.wav",
    ]:
        print("=" * 100)
        print("asr_task")
        print(f"{audio_path=}")

        start_time = perf_counter()
        output, _ = s2s_inference.run_infer(
            audio_path=audio_path,
            # message="Translate the speech to english.",
            # message="Translate the speech to text.",
            message="Convert the speech to text.",
            mode=None,
        )
        print(f"{output=}", flush=True)
        print(f"cost: {perf_counter() - start_time}")


def asr_text_stream():
    s2s_inference = S2SInference(
        model_path,
        audio_tokenizer_type,
        audio_tokenizer_model_path=audio_tokenizer_model_path,
        sense_voice_model_path=sense_voice_model_path,
        flow_path=None,
    )

    for audio_path in [
        # os.path.join(ASSETS_DIR, "asr_example_zh.wav"),
        "/VITA-Audio/asset/介绍一下上海.wav",
        # "/VITA-Audio/asset/发表一个悲伤的演讲.wav",
        # "/VITA-Audio/asset/发表一个振奋人心的演讲.wav",
    ]:
        print("=" * 100)
        print("asr_text_stream_task")
        print(f"{audio_path=}")

        times = []
        start_time = perf_counter()
        generated_text = ""
        for new_text in s2s_inference.run_infer_stream(
            audio_path=audio_path,
            # message="Translate the speech to text.",
            message="Convert the speech to text.",
            mode=None,
        ):
            times.append(perf_counter() - start_time)
            generated_text += new_text
            print(new_text, end=" ", flush=True)
            start_time = perf_counter()
        print(
            f"\ngenerate [{generated_text}] first token cost time: {times[0]} s, {len(times)} tokens cost time: {sum(times)} s\n"
        )


def asr_live():
    """
    need ft to support speech streaming asr
    see: https://github.com/FunAudioLLM/SenseVoice small
    """
    pass


# ==============================================================
# TTS
def tts():
    import torchaudio

    s2s_inference = S2SInference(
        model_path,
        audio_tokenizer_type,
        flow_path=flow_path,
    )

    # bad case: use glm4voice tokenizer to generate tts with Boost/Balance mtp llm(7B)
    TTS_texts = [
        # "我们将为全球城市的可持续发展贡献力量。",
        # "通天河 灵感大王",
        # "他本是我莲花池里养大的金鱼，每日浮头听经，修成手段。那一柄九瓣铜锤，乃是一枝未开的菡萏，被他运炼成兵。不知是那一日，海潮泛涨，走到此间。我今早扶栏看花，却不见这厮出拜，掐指巡纹，算着他在此成精，害你师父，故此未及梳妆，运神功，织个竹篮儿擒他。",
        # "一二三四五六七八九十",
        # "One Two Tree Four Five Six Seven Eight Night Ten",
        # "1 2 3 4 5 6 7 8 9 10",
        # "12345678910",
        # "两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。",
        # "坡上立着一只鹅，坡下就是一条河。宽宽的河，肥肥的鹅，鹅要过河，河要渡鹅不知是鹅过河，还是河渡鹅?",
        # "扁担长，板凳宽，扁担没有板凳宽，板凳没有扁担长。扁担绑在板凳上，板凳不让扁担绑在板凳上。",
        # "化肥会挥发，黑化肥发灰，灰化肥发黑。黑化肥发灰会挥发；灰化肥挥发会发黑。黑化肥挥发发灰会花飞；灰化肥挥发发黑会飞花，黑灰化肥会挥发发灰黑讳为花飞；灰黑化肥会挥发发黑灰为讳飞花。",
        # "圆桌儿、方桌儿没有腿儿，墨水瓶儿里没有水儿，花瓶里有花儿没有叶儿，练习本儿上写字儿没有准儿，甘蔗好吃净是节儿。西瓜挺大没有味儿，坛儿里的小米儿长了虫儿，鸡毛掸子成了棍儿，水缸沿儿上系围裙儿，耗子打更猫打盹儿，新买的小褂儿没钉扣儿，奶奶想说没有劲儿。",
        "起床歌：小宝宝，起得早，睁开眼，眯眯笑，咿呀呀，学说话，伸伸手，要人抱。穿衣歌小胳膊，穿袖子，穿上衣，扣扣子，小脚丫，穿裤子，穿上袜子穿鞋子。小镜子-小镜子，圆又圆，看宝宝，露笑脸。闭上眼，做个梦，变月亮，挂上天。小铃铛叮铃铃，叮铃铃，一会远，一会近。小宝宝，耳朵灵，听铃声，找到铃。学画画小宝宝，学画画，大蜡笔，手中拿，画小鸭，叫嘎嘎，画小马，骑回家。大鞋子大鞋子，像只船，爸爸穿，我也穿，一二一，向前走，走呀走，翻了船。逛公园逛公园，宝宝笑，东看看，西瞧瞧，花儿香，鸟儿叫，小草绿，小树摇。看画报小娃娃，看画报，睁大眼，仔细瞧，布娃娃，哈哈笑，伸伸手，要你抱。搭积木大积木，红黄兰，小宝宝，最爱玩，搭火车，钻山洞，盖高楼，连着天。小汽车小汽车，嘀嘀嘀，开过来，开过去，小宝宝，当司机，送妈妈，上班去。藏猫猫儿歌：躲猫猫，躲猫猫， 猫猫、猫猫在哪里？喵……猫咪在这里。",
    ]

    for text in TTS_texts:
        print("=" * 100)
        print("tts_task")
        print(f"{text=}")

        start = perf_counter()
        output, tts_speech = s2s_inference.run_infer(
            message="Convert the text to speech.\n" + text,
            mode=None,
            do_sample=True,
        )
        print(f"{output=}", flush=True)
        print(f"{tts_speech.shape=}", flush=True)
        print(f"cost: {perf_counter()-start}")

        wav_path = os.path.join(ASSETS_DIR, text[:16] + ".wav")
        print(f"save to {wav_path}")
        torchaudio.save(wav_path, tts_speech.unsqueeze(0), 22050, format="wav")


# ==============================================================
# Clone TTS
def tts_clone():
    import torchaudio

    s2s_inference = S2SInference(
        model_path,
        audio_tokenizer_type,
        audio_tokenizer_model_path=audio_tokenizer_model_path,
        sense_voice_model_path=sense_voice_model_path,
        flow_path=flow_path,
    )

    TTS_texts = [
        "我们将为全球城市的可持续发展贡献力量。",
        # "通天河 灵感大王",
        # "他本是我莲花池里养大的金鱼，每日浮头听经，修成手段。那一柄九瓣铜锤，乃是一枝未开的菡萏，被他运炼成兵。不知是那一日，海潮泛涨，走到此间。我今早扶栏看花，却不见这厮出拜，掐指巡纹，算着他在此成精，害你师父，故此未及梳妆，运神功，织个竹篮儿擒他。",
        # "一二三四五六七八九十",
        # "One Two Tree Four Five Six Seven Eight Night Ten",
        # "1 2 3 4 5 6 7 8 9 10",
        # "12345678910",
        # "两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。",
        # "坡上立着一只鹅，坡下就是一条河。宽宽的河，肥肥的鹅，鹅要过河，河要渡鹅不知是鹅过河，还是河渡鹅?",
        # "扁担长，板凳宽，扁担没有板凳宽，板凳没有扁担长。扁担绑在板凳上，板凳不让扁担绑在板凳上。",
        # "化肥会挥发，黑化肥发灰，灰化肥发黑。黑化肥发灰会挥发；灰化肥挥发会发黑。黑化肥挥发发灰会花飞；灰化肥挥发发黑会飞花，黑灰化肥会挥发发灰黑讳为花飞；灰黑化肥会挥发发黑灰为讳飞花。",
        # "圆桌儿、方桌儿没有腿儿，墨水瓶儿里没有水儿，花瓶里有花儿没有叶儿，练习本儿上写字儿没有准儿，甘蔗好吃净是节儿。西瓜挺大没有味儿，坛儿里的小米儿长了虫儿，鸡毛掸子成了棍儿，水缸沿儿上系围裙儿，耗子打更猫打盹儿，新买的小褂儿没钉扣儿，奶奶想说没有劲儿。",
        # "起床歌：小宝宝，起得早，睁开眼，眯眯笑，咿呀呀，学说话，伸伸手，要人抱。穿衣歌小胳膊，穿袖子，穿上衣，扣扣子，小脚丫，穿裤子，穿上袜子穿鞋子。小镜子-小镜子，圆又圆，看宝宝，露笑脸。闭上眼，做个梦，变月亮，挂上天。小铃铛叮铃铃，叮铃铃，一会远，一会近。小宝宝，耳朵灵，听铃声，找到铃。学画画小宝宝，学画画，大蜡笔，手中拿，画小鸭，叫嘎嘎，画小马，骑回家。大鞋子大鞋子，像只船，爸爸穿，我也穿，一二一，向前走，走呀走，翻了船。逛公园逛公园，宝宝笑，东看看，西瞧瞧，花儿香，鸟儿叫，小草绿，小树摇。看画报小娃娃，看画报，睁大眼，仔细瞧，布娃娃，哈哈笑，伸伸手，要你抱。搭积木大积木，红黄兰，小宝宝，最爱玩，搭火车，钻山洞，盖高楼，连着天。小汽车小汽车，嘀嘀嘀，开过来，开过去，小宝宝，当司机，送妈妈，上班去。藏猫猫儿歌：躲猫猫，躲猫猫， 猫猫、猫猫在哪里？喵……猫咪在这里。",
    ]

    for prompt_audio_path in [
        # "/VITA-Audio/asset/2631296891109983590.wav",
        # "/VITA-Audio/asset/379838640-d5ff0815-74f8-4738-b0f1-477cfc8dcc2d.wav",
        "/VITA-Audio/asset/4202818730519913143.wav",
    ]:
        subprocess.run(f"cp {prompt_audio_path} {ASSETS_DIR}", shell=True)
        file_name = os.path.basename(prompt_audio_path)[:-4]
        for text in TTS_texts:
            print("=" * 100)
            print("clone_tts_task")
            print(f"{text=} {prompt_audio_path=}")

            start = perf_counter()
            output, tts_speech = s2s_inference.run_infer(
                prompt_audio_path=prompt_audio_path,
                # message="Translate the text to speech.\n" + text,
                message="Convert the text to speech.\n" + text,
                mode=None,
                do_sample=True,
            )
            print(f"{output=}", flush=True)
            print(f"{tts_speech.shape=}", flush=True)
            print(f"cost: {perf_counter()-start} s")

            wav_path = os.path.join(ASSETS_DIR, file_name[:16] + "_" + text[:16] + ".wav")
            print(f"save to {wav_path}")
            torchaudio.save(wav_path, tts_speech.unsqueeze(0), 22050, format="wav")


# ==============================================================
# TTS stream
def tts_stream():
    """all text gen over, then to gen speech"""
    import torchaudio
    import soundfile as sf

    s2s_inference = S2SInference(
        model_path,
        audio_tokenizer_type,
        flow_path=flow_path,
    )

    TTS_texts = [
        # "他本是我莲花池里养大的金鱼，每日浮头听经，修成手段。那一柄九瓣铜锤，乃是一枝未开的菡萏，被他运炼成兵。不知是那一日，海潮泛涨，走到此间。我今早扶栏看花，却不见这厮出拜，掐指巡纹，算着他在此成精，害你师父，故此未及梳妆，运神功，织个竹篮儿擒他。",  # warmup
        # "他本是我莲花池里养大的金鱼，每日浮头听经，修成手段。那一柄九瓣铜锤，乃是一枝未开的菡萏，被他运炼成兵。不知是那一日，海潮泛涨，走到此间。我今早扶栏看花，却不见这厮出拜，掐指巡纹，算着他在此成精，害你师父，故此未及梳妆，运神功，织个竹篮儿擒他。",
        # "起床歌：小宝宝，起得早，睁开眼，眯眯笑，咿呀呀，学说话，伸伸手，要人抱。穿衣歌小胳膊，穿袖子，穿上衣，扣扣子，小脚丫，穿裤子，穿上袜子穿鞋子。",
        "小镜子-小镜子，圆又圆，看宝宝，露笑脸。闭上眼，做个梦，变月亮，挂上天。",
        # "小铃铛叮铃铃，叮铃铃，一会远，一会近。小宝宝，耳朵灵，听铃声，找到铃。学画画小宝宝，学画画，大蜡笔，手中拿，画小鸭，叫嘎嘎，画小马，骑回家。大鞋子大鞋子，像只船，爸爸穿，我也穿，一二一，向前走，走呀走，翻了船。",
        # "逛公园逛公园，宝宝笑，东看看，西瞧瞧，花儿香，鸟儿叫，小草绿，小树摇。看画报小娃娃，看画报，睁大眼，仔细瞧，布娃娃，哈哈笑，伸伸手，要你抱。",
        # "搭积木大积木，红黄兰，小宝宝，最爱玩，搭火车，钻山洞，盖高楼，连着天。小汽车小汽车，嘀嘀嘀，开过来，开过去，小宝宝，当司机，送妈妈，上班去。藏猫猫儿歌：躲猫猫，躲猫猫， 猫猫、猫猫在哪里？喵……猫咪在这里。",
    ]

    for text in TTS_texts:
        print("=" * 100)
        print("tts_stream_task")
        print(f"{text=}")

        file_name = text[:16]
        times = []
        start_time = perf_counter()

        generated_text = ""
        audio_decode_time = []
        audios = []
        for new_text in s2s_inference.run_infer_stream(
            message="Convert the text to speech.\n" + text,
            mode=None,
            do_sample=True,
        ):
            times.append(perf_counter() - start_time)
            generated_text += new_text
            print(new_text, end=" ", flush=True)
            start_time = perf_counter()
        print(
            f"\ngenerate first token cost time: {times[0]} s, {len(times)} tokens cost time: {sum(times)} s\n"
        )

        audio_decode_time = []
        audios = []
        audio_segments = find_audio_segments_regex(generated_text)
        for audio_idx, audio_segment in enumerate(audio_segments):
            start = perf_counter()

            audio_tokens = extract_token_ids_as_int(audio_segment)
            print(f"{audio_tokens=}")

            tts_speech = s2s_inference.audio_tokenizer.decode(audio_tokens)
            audio_decode_time.append(perf_counter() - start)

            # wav_path = os.path.join(ASSETS_DIR, f"qa_stream_{audio_idx}_{file_name}")
            # print(f"save to {wav_path=}")
            # torchaudio.save(wav_path, tts_speech.unsqueeze(0), 22050, format="wav")

            audios.append(tts_speech.cpu().numpy())

        wav_path = os.path.join(ASSETS_DIR, f"sts_stream_{file_name}.wav")
        print(f"save to {wav_path=}")
        sf.write(wav_path, np.concatenate(audios), samplerate=22050)
        info = sf.info(wav_path, verbose=True)

        print(
            f"\ngenerate first audio segment cost time: {audio_decode_time[0]} s, {len(audio_decode_time)} segment cost time: {sum(audio_decode_time)} s | wav duration: {info.duration} s | audio tokenizer decode RTF: {sum(audio_decode_time)/info.duration} \n"
        )


# TTS stream
def tts_text_stream():
    import torchaudio
    import soundfile as sf

    s2s_inference = S2SInference(
        model_path,
        audio_tokenizer_type,
        flow_path=flow_path,
    )

    TTS_texts = [
        "你好，hello.",
        # "他本是我莲花池里养大的金鱼，每日浮头听经，修成手段。那一柄九瓣铜锤，乃是一枝未开的菡萏，被他运炼成兵。不知是那一日，海潮泛涨，走到此间。我今早扶栏看花，却不见这厮出拜，掐指巡纹，算着他在此成精，害你师父，故此未及梳妆，运神功，织个竹篮儿擒他。",  # warmup
        # "他本是我莲花池里养大的金鱼，每日浮头听经，修成手段。那一柄九瓣铜锤，乃是一枝未开的菡萏，被他运炼成兵。不知是那一日，海潮泛涨，走到此间。我今早扶栏看花，却不见这厮出拜，掐指巡纹，算着他在此成精，害你师父，故此未及梳妆，运神功，织个竹篮儿擒他。",
        # "起床歌：小宝宝，起得早，睁开眼，眯眯笑，咿呀呀，学说话，伸伸手，要人抱。穿衣歌小胳膊，穿袖子，穿上衣，扣扣子，小脚丫，穿裤子，穿上袜子穿鞋子。",
        # "小镜子-小镜子，圆又圆，看宝宝，露笑脸。闭上眼，做个梦，变月亮，挂上天。",
        # "小铃铛叮铃铃，叮铃铃，一会远，一会近。小宝宝，耳朵灵，听铃声，找到铃。学画画小宝宝，学画画，大蜡笔，手中拿，画小鸭，叫嘎嘎，画小马，骑回家。大鞋子大鞋子，像只船，爸爸穿，我也穿，一二一，向前走，走呀走，翻了船。",
        # "逛公园逛公园，宝宝笑，东看看，西瞧瞧，花儿香，鸟儿叫，小草绿，小树摇。看画报小娃娃，看画报，睁大眼，仔细瞧，布娃娃，哈哈笑，伸伸手，要你抱。",
        # "搭积木大积木，红黄兰，小宝宝，最爱玩，搭火车，钻山洞，盖高楼，连着天。小汽车小汽车，嘀嘀嘀，开过来，开过去，小宝宝，当司机，送妈妈，上班去。藏猫猫儿歌：躲猫猫，躲猫猫， 猫猫、猫猫在哪里？喵……猫咪在这里。",
    ]

    for text in TTS_texts:
        print("=" * 100)
        print("tts_stream_task")
        print(f"{text=}")

        chunk_size = 4 * 2

        file_name = text[:16]
        times = []
        start_time = perf_counter()

        generated_text = ""
        sub_generated_text = ""
        audio_decode_time = []
        audios = []
        for new_text in s2s_inference.run_infer_stream(
            message="Convert the text to speech.\n" + text,
            mode=None,
            do_sample=True,
        ):
            times.append(perf_counter() - start_time)
            generated_text += new_text
            sub_generated_text += new_text
            print(new_text, end="", flush=True)
            if "<|end_of_audio|>" in new_text:
                audio_segments = find_audio_segments_regex(sub_generated_text)
                for audio_idx, audio_segment in enumerate(audio_segments):
                    audio_tokens = extract_token_ids_as_int(audio_segment)
                    print(f"\n{audio_tokens=}", flush=True)
                    start_time = perf_counter()
                    tts_speech = s2s_inference.audio_tokenizer.decode(audio_tokens)
                    audio_decode_time.append(perf_counter() - start_time)

                    # wav_path = os.path.join(output_dir, file_name + f"_{pos}.wav")
                    # torchaudio.save(wav_path, tts_speech.unsqueeze(0), 22050, format="wav")

                    audios.append(tts_speech.cpu().numpy())
                sub_generated_text = ""
            start_time = perf_counter()
        print(
            f"\ngenerate first token cost time: {times[0]} s, {len(times)} tokens cost time: {sum(times)} s\n"
        )

        wav_path = os.path.join(ASSETS_DIR, f"sts_stream_{file_name}.wav")
        print(f"save to {wav_path=}")
        sf.write(wav_path, np.concatenate(audios), samplerate=22050)
        info = sf.info(wav_path, verbose=True)

        print(
            f"\ngenerate first audio segment cost time: {audio_decode_time[0]} s, {len(audio_decode_time)} segment cost time: {sum(audio_decode_time)} s | wav duration: {info.duration} s | audio tokenizer decode RTF: {sum(audio_decode_time)/info.duration} \n"
        )


# TTS stream
def tts_audio_chunk_static_stream():
    import torchaudio
    import soundfile as sf

    s2s_inference = S2SInference(
        model_path,
        audio_tokenizer_type,
        flow_path=flow_path,
    )

    TTS_texts = [
        # "他本是我莲花池里养大的金鱼，每日浮头听经，修成手段。那一柄九瓣铜锤，乃是一枝未开的菡萏，被他运炼成兵。不知是那一日，海潮泛涨，走到此间。我今早扶栏看花，却不见这厮出拜，掐指巡纹，算着他在此成精，害你师父，故此未及梳妆，运神功，织个竹篮儿擒他。",  # warmup
        # "他本是我莲花池里养大的金鱼，每日浮头听经，修成手段。那一柄九瓣铜锤，乃是一枝未开的菡萏，被他运炼成兵。不知是那一日，海潮泛涨，走到此间。我今早扶栏看花，却不见这厮出拜，掐指巡纹，算着他在此成精，害你师父，故此未及梳妆，运神功，织个竹篮儿擒他。",
        # "起床歌：小宝宝，起得早，睁开眼，眯眯笑，咿呀呀，学说话，伸伸手，要人抱。穿衣歌小胳膊，穿袖子，穿上衣，扣扣子，小脚丫，穿裤子，穿上袜子穿鞋子。",
        "小镜子-小镜子，圆又圆，看宝宝，露笑脸。闭上眼，做个梦，变月亮，挂上天。",
        # "小铃铛叮铃铃，叮铃铃，一会远，一会近。小宝宝，耳朵灵，听铃声，找到铃。学画画小宝宝，学画画，大蜡笔，手中拿，画小鸭，叫嘎嘎，画小马，骑回家。大鞋子大鞋子，像只船，爸爸穿，我也穿，一二一，向前走，走呀走，翻了船。",
        # "逛公园逛公园，宝宝笑，东看看，西瞧瞧，花儿香，鸟儿叫，小草绿，小树摇。看画报小娃娃，看画报，睁大眼，仔细瞧，布娃娃，哈哈笑，伸伸手，要你抱。",
        # "搭积木大积木，红黄兰，小宝宝，最爱玩，搭火车，钻山洞，盖高楼，连着天。小汽车小汽车，嘀嘀嘀，开过来，开过去，小宝宝，当司机，送妈妈，上班去。藏猫猫儿歌：躲猫猫，躲猫猫， 猫猫、猫猫在哪里？喵……猫咪在这里。",
    ]

    for i, text in enumerate(TTS_texts):
        print("=" * 100)
        print("tts_stream_task")
        print(f"{text=}")

        chunk_size = 4 * 2
        session_id = i

        file_name = text[:16]
        times = []
        start_time = perf_counter()

        generated_text = ""
        raw_text = ""
        audio_decode_time = []
        audio_chunk = []
        audios = []
        for new_text in s2s_inference.run_infer_stream(
            message="Convert the text to speech.\n" + text,
            mode=None,
            do_sample=True,
        ):
            times.append(perf_counter() - start_time)
            generated_text += new_text
            print(new_text, end="", flush=True)
            if "audio" not in new_text:
                raw_text += new_text
                continue
            audio_tokens = extract_token_ids_as_int(new_text)
            print(f"\n{audio_tokens=}", flush=True)
            if not audio_tokens:
                continue
            audio_chunk.extend(audio_tokens)
            if len(audio_chunk) % chunk_size == 0:
                chunk = audio_chunk
                print(f"\n{chunk=}", flush=True)
                start_time = perf_counter()
                tts_speech = s2s_inference.audio_tokenizer.decode(chunk, session_id=session_id)
                audio_decode_time.append(perf_counter() - start_time)

                audios.append(tts_speech.cpu().numpy())
                audio_chunk = []
                start_time = perf_counter()

        if len(audio_chunk) > 0:
            chunk = audio_chunk
            print(f"\nlast {chunk=}", flush=True)
            start_time = perf_counter()
            tts_speech = s2s_inference.audio_tokenizer.decode(chunk, session_id=session_id)
            audio_decode_time.append(perf_counter() - start_time)

            audios.append(tts_speech.cpu().numpy())

        print(
            f"\ngenerate [{raw_text}] first token cost time: {times[0]} s, {len(times)} tokens cost time: {sum(times)} s\n"
        )

        wav_path = os.path.join(ASSETS_DIR, f"sts_stream_{file_name}.wav")
        print(f"save to {wav_path=}")
        sf.write(wav_path, np.concatenate(audios), samplerate=22050)
        info = sf.info(wav_path, verbose=True)

        print(
            f"\ngenerate first audio segment cost time: {audio_decode_time[0]} s, {len(audio_decode_time)} segment cost time: {sum(audio_decode_time)} s | wav duration: {info.duration} s | audio tokenizer decode RTF: {sum(audio_decode_time)/info.duration} \n"
        )


def tts_audio_chunk_dynamic_stream():
    """
    smooth chunk tokens_decode_out from glm4voice: chunk_size_list = [25, 50, 100, 150, 200]
    """
    import torchaudio
    import soundfile as sf

    s2s_inference = S2SInference(
        model_path,
        audio_tokenizer_type,
        flow_path=flow_path,
    )

    TTS_texts = [
        # "你好，hello.",
        "他本是我莲花池里养大的金鱼，每日浮头听经，修成手段。那一柄九瓣铜锤，乃是一枝未开的菡萏，被他运炼成兵。不知是那一日，海潮泛涨，走到此间。我今早扶栏看花，却不见这厮出拜，掐指巡纹，算着他在此成精，害你师父，故此未及梳妆，运神功，织个竹篮儿擒他。",  # warmup
        # "他本是我莲花池里养大的金鱼，每日浮头听经，修成手段。那一柄九瓣铜锤，乃是一枝未开的菡萏，被他运炼成兵。不知是那一日，海潮泛涨，走到此间。我今早扶栏看花，却不见这厮出拜，掐指巡纹，算着他在此成精，害你师父，故此未及梳妆，运神功，织个竹篮儿擒他。",
        # "起床歌：小宝宝，起得早，睁开眼，眯眯笑，咿呀呀，学说话，伸伸手，要人抱。穿衣歌小胳膊，穿袖子，穿上衣，扣扣子，小脚丫，穿裤子，穿上袜子穿鞋子。",
        # "小镜子-小镜子，圆又圆，看宝宝，露笑脸。闭上眼，做个梦，变月亮，挂上天。",
        # "小铃铛叮铃铃，叮铃铃，一会远，一会近。小宝宝，耳朵灵，听铃声，找到铃。学画画小宝宝，学画画，大蜡笔，手中拿，画小鸭，叫嘎嘎，画小马，骑回家。大鞋子大鞋子，像只船，爸爸穿，我也穿，一二一，向前走，走呀走，翻了船。",
        # "逛公园逛公园，宝宝笑，东看看，西瞧瞧，花儿香，鸟儿叫，小草绿，小树摇。看画报小娃娃，看画报，睁大眼，仔细瞧，布娃娃，哈哈笑，伸伸手，要你抱。",
        # "搭积木大积木，红黄兰，小宝宝，最爱玩，搭火车，钻山洞，盖高楼，连着天。小汽车小汽车，嘀嘀嘀，开过来，开过去，小宝宝，当司机，送妈妈，上班去。藏猫猫儿歌：躲猫猫，躲猫猫， 猫猫、猫猫在哪里？喵……猫咪在这里。",
    ]

    for i, text in enumerate(TTS_texts):
        print("=" * 100)
        print("tts_stream_task")
        print(f"{text=}")

        prompt_speech_feat = torch.zeros(1, 0, 80).to("cuda")
        flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to("cuda")

        this_uuid = str(uuid.uuid4())
        tts_mels = []
        prev_mel = None

        is_finalize = False
        chunk_size_list = [8, 16, 25, 50, 100, 150, 200]  # like tcp cwnd
        # chunk_size_list = [8]  # like tcp cwnd
        chunk_size_idx = 0
        chunk_size = chunk_size_list[chunk_size_idx]

        file_name = text[:16]
        times = []
        start_time = perf_counter()

        generated_text = ""
        raw_text = ""
        audio_decode_time = []
        audio_chunk = []
        audios = []
        for new_text in s2s_inference.run_infer_stream(
            message="Convert the text to speech.\n" + text,
            mode=None,
            do_sample=True,
        ):
            times.append(perf_counter() - start_time)
            generated_text += new_text
            print(f"{new_text=}", flush=True)
            if "<|begin_of_audio|>" in new_text:
                new_text = new_text.replace("<|begin_of_audio|>", "")
            if "<|end_of_audio|>" in new_text:
                new_text = new_text.replace("<|end_of_audio|>", "")
            if "<|im_end|>" in new_text:
                new_text = new_text.replace("<|im_end|>", "")
                is_finalize = True
            audio_tokens = extract_token_ids_as_int(new_text)
            print(f"\n{audio_tokens=}", flush=True)
            if not audio_tokens and is_finalize is False:
                raw_text += new_text
                continue
            audio_chunk.extend(audio_tokens)
            print(f"{is_finalize=} {len(audio_chunk)=}")
            if len(audio_chunk) >= chunk_size or (is_finalize and audio_chunk):
                print(f"\n{audio_chunk=}", flush=True)
                if chunk_size_idx < len(chunk_size_list) - 1:
                    chunk_size_idx += 1
                    chunk_size = chunk_size_list[chunk_size_idx]
                tts_token = torch.tensor(audio_chunk, device="cuda").unsqueeze(0)

                if prev_mel is not None:
                    prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)

                # gen waveform and mel-spectrogram feat
                start_time = perf_counter()
                tts_speech, tts_mel = s2s_inference.audio_tokenizer.audio_decoder.token2wav(
                    tts_token,
                    uuid=this_uuid,
                    prompt_token=flow_prompt_speech_token.to("cuda"),
                    prompt_feat=prompt_speech_feat.to("cuda"),
                    finalize=is_finalize,
                )
                audio_decode_time.append(perf_counter() - start_time)
                prev_mel = tts_mel

                print(tts_speech.shape)
                audios.append(tts_speech.squeeze(0).cpu().numpy())  # T
                tts_mels.append(tts_mel)
                flow_prompt_speech_token = torch.cat((flow_prompt_speech_token, tts_token), dim=-1)
                audio_chunk = []
                start_time = perf_counter()

        print(
            f"\ngenerate [{raw_text}] first token cost time: {times[0]} s, {len(times)} tokens cost time: {sum(times)} s\n"
        )
        print(f"{chunk_size_list=}")
        print(f"{audio_decode_time=}")

        wav_path = os.path.join(ASSETS_DIR, f"sts_stream_{file_name}.wav")
        print(f"save to {wav_path=}")
        sf.write(wav_path, np.concatenate(audios), samplerate=22050)
        info = sf.info(wav_path, verbose=True)

        print(
            f"\ngenerate first audio segment cost time: {audio_decode_time[0]} s, {len(audio_decode_time)} segment cost time: {sum(audio_decode_time)} s | wav duration: {info.duration} s | audio tokenizer decode RTF: {sum(audio_decode_time)/info.duration} \n"
        )


def print_model_params(model: torch.nn.Module, extra_info=""):
    # print the number of parameters in the model
    model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(model)
    print(f"{extra_info} {model_million_params} M parameters")


def find_audio_segments_regex(text):
    """
    Find all substrings between <|begin_of_audio|> and <|end_of_audio|> using regex.

    Args:
        text (str): The input string to search through

    Returns:
        list: A list of all found audio segments (substrings between the delimiters)
    """
    pattern = re.compile(r"<\|begin_of_audio\|>(.*?)<\|end_of_audio\|>", re.DOTALL)
    segments = pattern.findall(text)
    return [segment.strip() for segment in segments]


def extract_token_ids_as_int(text):
    pattern = re.compile(r"<\|audio_(\d+)\|>")
    token_ids = pattern.findall(text)
    return [int(id) for id in token_ids]


def custom_init_weights(module):
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)
    elif isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d):
        torch.nn.init.constant_(module.weight, 1)
        torch.nn.init.constant_(module.bias, 0)


class TextAudioIteratorStreamer(TextIteratorStreamer):
    def __init__(
        self,
        tokenizer: "AutoTokenizer",
        skip_prompt: bool = False,
        timeout: Optional[float] = None,
        **decode_kwargs,
    ):
        super().__init__(tokenizer, skip_prompt, timeout, **decode_kwargs)

        # self.audio_offset = tokenizer.convert_tokens_to_ids("<|audio_0|>")
        self.audio_offset = tokenizer.convert_tokens_to_ids("<|begin_of_audio|>")
        self.num_decode_tokens = 0

    def put(self, value):
        """
        Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.
        """
        print(f"{value=}")
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        self.num_decode_tokens += len(value)

        # Add the new token to the cache and decodes the entire thing.
        self.token_cache.extend(value.tolist())
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

        # After the symbol for a new line, we flush the cache.
        if text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        # If the last token is a CJK character, we print the characters.
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        elif self.token_cache[-1] >= self.audio_offset:
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
        # which may change with the subsequent token -- there are probably smarter ways to do this!)
        else:
            printable_text = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable_text)

        self.on_finalized_text(printable_text)

        while self.text_queue.qsize() > 10:
            time.sleep(0.01)


class BenchmarkIteratorStreamer(TextIteratorStreamer):
    def __init__(
        self,
        tokenizer: "AutoTokenizer",
        skip_prompt: bool = False,
        timeout: Optional[float] = None,
        **decode_kwargs,
    ):
        super().__init__(tokenizer, skip_prompt, timeout, **decode_kwargs)

        self.num_decode_tokens = 0

    def put(self, value):
        """
        Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.
        """
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        self.num_decode_tokens += len(value)

        printable_text = " ".join([str(x) for x in value.tolist()]) + " "
        self.on_finalized_text(printable_text)


class S2SInference:
    def __init__(
        self,
        llm_model_path,
        audio_tokenizer_type,
        audio_tokenizer_model_path=None,
        sense_voice_model_path=None,
        flow_path=None,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        audio_tokenizer_rank=0,
    ):
        from evaluation.get_chat_template import qwen2_chat_template as chat_template

        add_generation_prompt = True

        default_system_message = []

        luke_system_message = [
            {
                "role": "system",
                "content": "Your Name: Luke\nYour Gender: male\n\nRespond in a text-audio interleaved manner.",
            },
        ]

        tokenizer = AutoTokenizer.from_pretrained(
            llm_model_path,
            trust_remote_code=True,
            chat_template=chat_template,
        )

        # https://huggingface.co/VITA-MLLM/VITA-Audio-Boost/blob/main/modeling_qwen2.py#L781
        # https://huggingface.co/VITA-MLLM/VITA-Audio-Balance/blob/main/modeling_qwen2.py#L781
        # https://huggingface.co/VITA-MLLM/VITA-Audio-Plus-Vanilla/blob/main/modeling_qwen2.py#L834
        model = AutoModelForCausalLM.from_pretrained(
            llm_model_path,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
        ).eval()
        # print("model", model)
        print(f"{model.config=}")
        # print(f"{model.config.model_type=}")
        print(f"{model.hf_device_map=}")

        model.generation_config = GenerationConfig.from_pretrained(
            llm_model_path, trust_remote_code=True
        )

        model.generation_config.max_new_tokens = 8192
        model.generation_config.chat_format = "chatml"
        model.generation_config.max_window_size = 8192
        model.generation_config.use_cache = True
        # model.generation_config.use_cache = False
        model.generation_config.do_sample = False
        model.generation_config.temperature = 1.0
        model.generation_config.top_k = 50
        model.generation_config.top_p = 1.0
        model.generation_config.num_beams = 1
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        print(f"{model.generation_config=}")

        audio_tokenizer = get_audio_tokenizer(
            model_type=audio_tokenizer_type,
            flow_path=flow_path,
            model_name_or_path=audio_tokenizer_model_path,
            rank=audio_tokenizer_rank,
            sense_voice_model_path=sense_voice_model_path,
        )
        audio_tokenizer.load_model()

        self.model = model
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.add_generation_prompt = add_generation_prompt
        self.default_system_message = default_system_message
        self.luke_system_message = luke_system_message

        self.audio_0_id = tokenizer("<|audio_0|>").input_ids[0]
        print(f"{self.audio_0_id=}")

    def benchmark_forward(self, mtp_inference_mode):
        print("-" * 100)
        print("benchmark_forward...")
        print(f"{mtp_inference_mode=}")

        total_time = 0

        past_key_values = None
        use_cache = True

        self.model.input_ids = None
        self.model.inputs_embeds = None
        self.model.hidden_states = [None] * (self.model.config.num_nextn_predict_layers + 1)
        self.model.position_ids = None
        self.model.attention_mask = None
        self.model.mtp_idx = -1
        self.model.num_prefill_tokens = -1

        model_max_length = 1024
        if mtp_inference_mode is not None:
            ori_mtp_inference_mode = self.model.generation_config.mtp_inference_mode
            self.model._prepare_mtp_for_generation(mtp_inference_mode, model_max_length)

        else:
            self.model._prepare_mtp_for_generation(
                self.model.generation_config.mtp_inference_mode, model_max_length
            )

        for i in tqdm.tqdm(range(1, model_max_length + 1)):
            if use_cache:
                input_ids = torch.tensor([i - 1], dtype=torch.long).unsqueeze(0).to("cuda")
                position_ids = torch.tensor([i - 1], dtype=torch.long).unsqueeze(0).to("cuda")
            else:
                input_ids = torch.arange(i, dtype=torch.long).unsqueeze(0).to("cuda")
                position_ids = torch.arange(i, dtype=torch.long).unsqueeze(0).to("cuda")

            attention_mask = torch.tensor([1] * i, dtype=torch.float).unsqueeze(0).to("cuda")

            torch.cuda.synchronize()
            start = time.time()

            output = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                num_logits_to_keep=1,
            )

            torch.cuda.synchronize()
            end = time.time()

            total_time += end - start
            # print(f"{i=} {total_time=}")

            past_key_values = output.past_key_values

        print()
        print(f"{total_time=}")
        print(f"second/token {total_time/model_max_length=}")
        print(f"token/second {model_max_length/total_time=}")

        if mtp_inference_mode is not None:
            self.model.mtp_inference_mode = ori_mtp_inference_mode

    def benchmark_generate(self, mtp_inference_mode):
        self.model.apply(custom_init_weights)

        print("-" * 100)
        print("benchmark_generate...")
        print(f"{mtp_inference_mode=}")

        total_time = 0
        self.model.generation_config.use_cache = True

        self.model.generation_config.max_new_tokens = 8192

        if mtp_inference_mode is not None:
            ori_mtp_inference_mode = self.model.generation_config.mtp_inference_mode
            self.model.generation_config.mtp_inference_mode = mtp_inference_mode

        input_ids = torch.tensor([0], dtype=torch.long).unsqueeze(0).to("cuda")

        torch.cuda.synchronize()
        start = time.time()

        output = self.model.generate(
            input_ids,
        )
        # print(f"{output.size()=}")

        torch.cuda.synchronize()
        end = time.time()

        total_time += end - start

        print()
        print(f"{total_time=}")
        print(f"second/token {total_time/output.size(1)=}")
        print(f"token/second {output.size(1)/total_time=}")

        if mtp_inference_mode is not None:
            self.model.generation_config.mtp_inference_mode = ori_mtp_inference_mode

    def benchmark_generate_stream(self, mtp_inference_mode):
        print("-" * 100)
        print("benchmark_generate_stream...")
        print(f"{mtp_inference_mode=}")

        self.model.apply(custom_init_weights)

        total_time = 0
        self.model.generation_config.use_cache = True

        # model_max_length = 8192
        model_max_length = 4096
        # model_max_length = 2048
        # model_max_length = 1024
        num_prefill_tokens = 32

        self.model.generation_config.max_new_tokens = model_max_length
        self.model.generation_config.do_sample = False

        if mtp_inference_mode is not None:
            ori_mtp_inference_mode = self.model.generation_config.mtp_inference_mode
            self.model.generation_config.mtp_inference_mode = mtp_inference_mode

        input_ids = torch.tensor([0] * num_prefill_tokens, dtype=torch.long).unsqueeze(0).to("cuda")

        streamer = BenchmarkIteratorStreamer(self.tokenizer, skip_prompt=True)
        generation_kwargs = dict(input_ids=input_ids, streamer=streamer)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)

        token_decode_time = []

        torch.cuda.synchronize()
        start = time.time()
        thread.start()

        generated_text = ""
        for new_text in tqdm.tqdm(streamer, total=model_max_length):
            generated_text += new_text
            end = time.time()

            token_decode_time.append(end - start)

            yield new_text

        # print(f"{len(generated_text)}")

        torch.cuda.synchronize()
        end = time.time()

        total_time += end - start

        print()
        print(f"{token_decode_time[-1]=}")
        print(f"{streamer.num_decode_tokens=}")
        print(f"second/token {token_decode_time[-1]/streamer.num_decode_tokens=}")
        print(f"token/second {streamer.num_decode_tokens/token_decode_time[-1]=}")

        # if mtp_inference_mode is None:
        #     mtp_inference_mode = []
        # with open(f'token_decode_time_{str(mtp_inference_mode)}.json', 'w') as f:
        #     json.dump(token_decode_time, f)

        if mtp_inference_mode is not None:
            self.model.generation_config.mtp_inference_mode = ori_mtp_inference_mode

    @torch.inference_mode()
    def run_infer(
        self,
        audio_path=None,
        prompt_audio_path=None,
        chunk_size=4,
        max_returned_tokens=4096,
        sample_rate=16000,
        request_id="",
        audio_feats=None,
        message="",
        use_past=False,
        mode="luke",
        do_sample=False,
        mtp_inference_mode=None,
    ):
        AUD_TAG_TOKEN = "<|audio|>"
        AUD_CONTEXT_TOKEN = "<|context_of_audio|>"
        AUD_START_TOKEN = "<|begin_of_audio|>"
        AUD_END_TOKEN = "<|end_of_audio|>"

        if prompt_audio_path is not None:
            system_message = [
                {
                    "role": "system",
                    "content": f"Your Voice: <|audio|>\n",
                },
            ]

        elif mode == "luke":
            system_message = self.luke_system_message

        else:
            system_message = self.default_system_message

        if prompt_audio_path is not None and self.audio_tokenizer.apply_to_role(
            "user", is_discrete=True
        ):
            print("discrete codec")
            # discrete codec
            audio_tokens = self.audio_tokenizer.encode(
                prompt_audio_path, is_discrete=True, is_contiguous=False
            )
            audio_tokens = "".join(f"<|audio_{i}|>" for i in audio_tokens)
            system_message[-1]["content"] = system_message[-1]["content"].replace(
                "<|audio|>", f"<|begin_of_audio|>{audio_tokens}<|end_of_audio|>"
            )

        if audio_path is not None:
            messages = system_message + [
                {
                    "role": "user",
                    "content": message + "\n<|audio|>",
                },
            ]
        else:
            messages = system_message + [
                {
                    "role": "user",
                    "content": message,
                },
            ]

        if audio_path is not None and self.audio_tokenizer.apply_to_role("user", is_discrete=True):
            print("discrete codec")
            # discrete codec
            audio_tokens = self.audio_tokenizer.encode(
                audio_path, is_discrete=True, is_contiguous=False
            )
            audio_tokens = "".join(f"<|audio_{i}|>" for i in audio_tokens)
            messages[-1]["content"] = messages[-1]["content"].replace(
                "<|audio|>", f"<|begin_of_audio|>{audio_tokens}<|end_of_audio|>"
            )

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=self.add_generation_prompt,
        )

        if (
            audio_path is not None or prompt_audio_path is not None
        ) and self.audio_tokenizer.apply_to_role("user", is_contiguous=True):
            print("contiguous codec")
            # contiguous codec
            audio_paths = []
            if audio_path is not None:
                audio_paths.append(audio_path)
            if prompt_audio_path is not None:
                audio_paths.append(prompt_audio_path)
            input_ids, audios, audio_indices = add_audio_input_contiguous(
                input_ids, audio_paths, self.tokenizer, self.audio_tokenizer
            )
        else:
            audios = None
            audio_indices = None

        input_ids = torch.tensor([input_ids], dtype=torch.long).to("cuda")
        print("input", self.tokenizer.decode(input_ids[0], skip_special_tokens=False), flush=True)

        self.model.generation_config.do_sample = do_sample

        if mtp_inference_mode is not None:
            ori_mtp_inference_mode = self.model.generation_config.mtp_inference_mode
            self.model.generation_config.mtp_inference_mode = mtp_inference_mode

        outputs = self.model.generate(
            input_ids,
            audios=audios,
            audio_indices=audio_indices,
        )

        outputs = outputs[0][input_ids.shape[1] :]
        output = self.tokenizer.decode(outputs, skip_special_tokens=False)

        audio_offset = self.tokenizer.convert_tokens_to_ids("<|audio_0|>")

        audio_tokens = []
        for token_id in outputs:
            if token_id >= audio_offset:
                audio_tokens.append(token_id - audio_offset)

        if len(audio_tokens) > 0:
            tts_speech = self.audio_tokenizer.decode(
                audio_tokens, source_speech_16k=prompt_audio_path
            )

        else:
            tts_speech = None

        if mtp_inference_mode is not None:
            self.model.generation_config.mtp_inference_mode = ori_mtp_inference_mode

        return output, tts_speech

    @torch.inference_mode()
    def run_infer_stream(
        self,
        audio_path=None,
        prompt_audio_path=None,
        chunk_size=4,
        max_returned_tokens=4096,
        sample_rate=16000,
        request_id="",
        audio_feats=None,
        message="",
        use_past=False,
        mode="luke",
        do_sample=False,
        mtp_inference_mode=None,
    ):
        if prompt_audio_path is not None:
            system_message = [
                {
                    "role": "system",
                    "content": f"Your Voice: <|audio|>\n",
                },
            ]

        elif mode == "luke":
            system_message = self.luke_system_message

        else:
            system_message = self.default_system_message

        if prompt_audio_path is not None and self.audio_tokenizer.apply_to_role(
            "user", is_discrete=True
        ):
            # discrete codec
            audio_tokens = self.audio_tokenizer.encode(prompt_audio_path)
            audio_tokens = "".join(f"<|audio_{i}|>" for i in audio_tokens)
            system_message[-1]["content"] = system_message[-1]["content"].replace(
                "<|audio|>", f"<|begin_of_audio|>{audio_tokens}<|end_of_audio|>"
            )

        if audio_path is not None:
            messages = system_message + [
                {
                    "role": "user",
                    "content": message + "\n<|audio|>",
                },
            ]
        else:
            messages = system_message + [
                {
                    "role": "user",
                    "content": message,
                },
            ]

        if audio_path is not None and self.audio_tokenizer.apply_to_role("user", is_discrete=True):
            print("discrete codec")
            # discrete codec
            audio_tokens = self.audio_tokenizer.encode(audio_path)
            audio_tokens = "".join(f"<|audio_{i}|>" for i in audio_tokens)
            messages[-1]["content"] = messages[-1]["content"].replace(
                "<|audio|>", f"<|begin_of_audio|>{audio_tokens}<|end_of_audio|>"
            )

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=self.add_generation_prompt,
        )

        if (
            audio_path is not None or prompt_audio_path is not None
        ) and self.audio_tokenizer.apply_to_role("user", is_contiguous=True):
            print("contiguous codec")
            # contiguous codec
            audio_paths = []
            if audio_path is not None:
                audio_paths.append(audio_path)
            if prompt_audio_path is not None:
                audio_paths.append(prompt_audio_path)
            input_ids, audios, audio_indices = add_audio_input_contiguous(
                input_ids, audio_paths, self.tokenizer, self.audio_tokenizer
            )
        else:
            audios = None
            audio_indices = None

        input_ids = torch.tensor([input_ids], dtype=torch.long).to("cuda")

        print("input", self.tokenizer.decode(input_ids[0], skip_special_tokens=False), flush=True)
        attention_mask = torch.ones((1, input_ids.shape[1]), dtype=torch.int64).to("cuda")

        self.model.generation_config.do_sample = do_sample

        if mtp_inference_mode is not None:
            print(f"{mtp_inference_mode=}")
            ori_mtp_inference_mode = self.model.generation_config.mtp_inference_mode
            self.model.generation_config.mtp_inference_mode = mtp_inference_mode

        streamer = TextAudioIteratorStreamer(self.tokenizer, skip_prompt=True)
        generation_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audios=audios,
            audio_indices=audio_indices,
            streamer=streamer,
        )
        print(
            f"input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape} audio_indices: {audio_indices} audios: {audios}"
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)

        thread.start()

        # generated_text = ""
        for new_text in streamer:
            # generated_text += new_text

            yield new_text

        # torch.cuda.synchronize()

        if mtp_inference_mode is not None:
            self.model.generation_config.mtp_inference_mode = ori_mtp_inference_mode


"""
IMAGE_GPU=T4 modal run src/llm/transformers/vita_voice.py --task tokenize
IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task dump_model

# LLM: sensevoice(no use ctc)_qwen2_mtp(no mtp)
IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task text 
IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task text_audio
IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task text_stream
IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task text_audio_stream

# LLM: qwen2_mtp
AUDIO_TOKENIZER_TYPE=glm4voice IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task text 
AUDIO_TOKENIZER_TYPE=glm4voice IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task text_audio
AUDIO_TOKENIZER_TYPE=glm4voice IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task text_stream
AUDIO_TOKENIZER_TYPE=glm4voice IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task text_audio_stream
MTP_LLM_MODEL=VITA-MLLM/VITA-Audio-Boost AUDIO_TOKENIZER_TYPE=glm4voice IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task text 
MTP_LLM_MODEL=VITA-MLLM/VITA-Audio-Boost AUDIO_TOKENIZER_TYPE=glm4voice IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task text_stream

# LLM(Plus-Vanilla): sensevoice(no use ctc)_qwen2_mtp(no mtp) + AudioTokenizer: sensevoice_glm4voice(sensevoice WavFrontend, decoder)
IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task sts
IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task sts_stream
IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task asr
IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task asr_text_stream
IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task tts
IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task tts_stream
IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task tts_clone
IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task tts_audio_chunk_static_stream
IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task tts_audio_chunk_dynamic_stream

# LLM(Plus-Boost): sensevoice(no use ctc)_qwen2_mtp(no mtp) + AudioTokenizer: sensevoice_glm4voice(sensevoice WavFrontend, decoder)
MTP_LLM_MODEL=VITA-MLLM/VITA-Audio-Plus-Boost IMAGE_GPU=L40s modal run src/llm/transformers/vita_voice.py --task sts
MTP_LLM_MODEL=VITA-MLLM/VITA-Audio-Plus-Boost IMAGE_GPU=L40s modal run src/llm/transformers/vita_voice.py --task sts_stream
MTP_LLM_MODEL=VITA-MLLM/VITA-Audio-Plus-Boost IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task asr
MTP_LLM_MODEL=VITA-MLLM/VITA-Audio-Plus-Boost IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task asr_text_stream
MTP_LLM_MODEL=VITA-MLLM/VITA-Audio-Plus-Boost IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task tts
MTP_LLM_MODEL=VITA-MLLM/VITA-Audio-Plus-Boost IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task tts_stream
MTP_LLM_MODEL=VITA-MLLM/VITA-Audio-Plus-Boost IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task tts_clone
MTP_LLM_MODEL=VITA-MLLM/VITA-Audio-Plus-Boost IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task tts_audio_chunk_static_stream
MTP_LLM_MODEL=VITA-MLLM/VITA-Audio-Plus-Boost IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task tts_audio_chunk_dynamic_stream

# LLM(Boost/Balance): qwen2_mtp + AudioTokenizer: glm4voice(tokenizer, decoder)
AUDIO_TOKENIZER_TYPE=glm4voice IMAGE_GPU=L40s modal run src/llm/transformers/vita_voice.py --task sts
AUDIO_TOKENIZER_TYPE=glm4voice IMAGE_GPU=L40s modal run src/llm/transformers/vita_voice.py --task sts_stream
AUDIO_TOKENIZER_TYPE=glm4voice IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task asr
AUDIO_TOKENIZER_TYPE=glm4voice IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task asr_text_stream
AUDIO_TOKENIZER_TYPE=glm4voice IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task tts
AUDIO_TOKENIZER_TYPE=glm4voice IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task tts_stream
AUDIO_TOKENIZER_TYPE=glm4voice IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task tts_clone
AUDIO_TOKENIZER_TYPE=glm4voice IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task tts_audio_chunk_static_stream
AUDIO_TOKENIZER_TYPE=glm4voice IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task tts_audio_chunk_dynamic_stream

MTP_LLM_MODEL=VITA-MLLM/VITA-Audio-Boost AUDIO_TOKENIZER_TYPE=glm4voice IMAGE_GPU=L40s modal run src/llm/transformers/vita_voice.py --task sts
MTP_LLM_MODEL=VITA-MLLM/VITA-Audio-Boost AUDIO_TOKENIZER_TYPE=glm4voice IMAGE_GPU=L40s modal run src/llm/transformers/vita_voice.py --task sts_stream
MTP_LLM_MODEL=VITA-MLLM/VITA-Audio-Boost AUDIO_TOKENIZER_TYPE=glm4voice IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task asr
MTP_LLM_MODEL=VITA-MLLM/VITA-Audio-Boost AUDIO_TOKENIZER_TYPE=glm4voice IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task asr_text_stream
MTP_LLM_MODEL=VITA-MLLM/VITA-Audio-Boost AUDIO_TOKENIZER_TYPE=glm4voice IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task tts
MTP_LLM_MODEL=VITA-MLLM/VITA-Audio-Boost AUDIO_TOKENIZER_TYPE=glm4voice IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task tts_stream
MTP_LLM_MODEL=VITA-MLLM/VITA-Audio-Boost AUDIO_TOKENIZER_TYPE=glm4voice IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task tts_clone
MTP_LLM_MODEL=VITA-MLLM/VITA-Audio-Boost AUDIO_TOKENIZER_TYPE=glm4voice IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task tts_audio_chunk_static_stream
MTP_LLM_MODEL=VITA-MLLM/VITA-Audio-Boost AUDIO_TOKENIZER_TYPE=glm4voice IMAGE_GPU=L4 modal run src/llm/transformers/vita_voice.py --task tts_audio_chunk_dynamic_stream
"""


@app.local_entrypoint()
def main(task: str = "tokenize"):
    tasks = {
        "tokenize": tokenize,
        "dump_model": dump_model,
        "text": text,
        "text_stream": text_stream,
        "text_audio": text_audio,
        "text_audio_stream": text_audio_stream,
        "asr": asr,
        "asr_text_stream": asr_text_stream,
        "sts": sts,
        "sts_stream": sts_stream,
        "tts": tts,
        "tts_stream": tts_stream,
        "tts_clone": tts_clone,
        "tts_audio_chunk_static_stream": tts_audio_chunk_static_stream,
        "tts_audio_chunk_dynamic_stream": tts_audio_chunk_dynamic_stream,
        "benchmark_llm": benchmark_llm,
        # "benchmark_sts": benchmark_sts,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(tasks[task])
