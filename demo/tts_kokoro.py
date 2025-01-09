import glob
import os
import sys
import logging
import io
import json

import numpy as np
import requests
import torch
import sounddevice as sd

from src.common.types import MODELS_DIR

"""
# if not espeak-ng, brew update
brew install espeak-ng
huggingface-cli download hexgrad/Kokoro-82M --quiet --local-dir --exclude "*.onnx" ./models/Kokoro82M
touch models/__init__.py
"""
try:
    sys.path.insert(1, os.path.join(MODELS_DIR, "Kokoro82M"))
    from models.Kokoro82M.kokoro import generate
    from models.Kokoro82M.models import build_model
except ModuleNotFoundError as e:
    logging.error("In order to use kokoro-tts, you need to `pip install achatbot[tts_kokoro]`.")
    raise Exception(f"Missing module: {e}")

"""
brew install espeak-ng
# wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx -O ./models/kokoro-v0_19.onnx
# use export_pytorch_voices_to_json or download from github
# wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json -O ./models/kokoro-voices.json
"""
try:
    from kokoro_onnx import Kokoro
except ModuleNotFoundError as e:
    logging.error(
        "In order to use kokoro-tts with onnx, you need to `pip install achatbot[tts_onnx_kokoro]`."
    )
    raise Exception(f"Missing module: {e}")


def run_torch_kokoro(text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL = build_model(os.path.join(MODELS_DIR, "Kokoro82M/kokoro-v0_19.pth"), device)

    total_params = 0
    for key, model in MODEL.items():
        print(f"{key} Model: {model}")
        params = sum(p.numel() for p in model.parameters())
        total_params += params
        model_million_params = params / 1e6
        print(f"{key} Model has {model_million_params:.3f} million parameters")

    model_million_params = total_params / 1e6
    print(f"Model total has {model_million_params:.3f} million parameters")

    VOICE_NAME = [
        "af",  # Default voice is a 50-50 mix of Bella & Sarah
        "af_bella",
        "af_sarah",
        "am_adam",
        "am_michael",
        "bf_emma",
        "bf_isabella",
        "bm_george",
        "bm_lewis",
        "af_nicole",
        "af_sky",
    ][0]
    VOICEPACK = torch.load(
        os.path.join(MODELS_DIR, f"Kokoro82M/voices/{VOICE_NAME}.pt"), weights_only=True
    ).to(device)
    print(f"Loaded voice: {VOICE_NAME}")

    audio_samples, out_ps = generate(MODEL, text, VOICEPACK, lang=VOICE_NAME[0])
    sd.play(audio_samples, 24000)
    sd.wait()


def export_pytorch_voices_to_json():
    """
    export pytorch voices to json for onnx
    """
    voices_json = {}
    voices_pt_file_list = glob.glob(os.path.join(MODELS_DIR, "Kokoro82M/voices/*.pt"))
    # print(voices_pt_file_list)

    for voice_file in voices_pt_file_list:
        voice = os.path.splitext(os.path.basename(voice_file))[0]
        print(f"voice {voice} file: {voice_file}")
        voice_data: np.ndarray = torch.load(voice_file).numpy()
        voices_json[voice] = voice_data.tolist()

    voice_json_file = os.path.join(MODELS_DIR, "Kokoro82M/kokoro-voices.json")
    with open(voice_json_file, "w") as f:
        json.dump(voices_json, f, indent=4)


def run_onnx_kokoro(text):
    model_struct_stats_ckpt = os.path.join(MODELS_DIR, "Kokoro82M/kokoro-v0_19.onnx")
    voices_file = os.path.join(MODELS_DIR, "Kokoro82M/kokoro-voices.json")
    print(model_struct_stats_ckpt, voices_file)
    kokoro = Kokoro(
        model_struct_stats_ckpt,
        voices_file,
        espeak_ng_lib_path="/usr/local/Cellar/espeak-ng/1.52.0/lib/libespeak-ng.1.dylib",
    )
    audio_samples, sample_rate = kokoro.create(text, voice="af_sarah", speed=1.0, lang="en-us")
    sd.play(audio_samples, sample_rate)
    sd.wait()


if __name__ == "__main__":
    text = "How could I know? It's an unanswerable question. Like asking an unborn child if they'll lead a good life. They haven't even been born."

    # run_torch_kokoro(text)

    # kexport_pytorch_voices_to_json()
    run_onnx_kokoro(text)
