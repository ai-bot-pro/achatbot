import glob
import logging
import os
import sys
from typing import AsyncGenerator

import numpy as np

from src.common.types import MODELS_DIR, PYAUDIO_PAFLOAT32
from src.common.session import Session
from src.common.interface import ITts
from src.modules.speech.tts.base import BaseTTS
from src.types.speech.tts.kokoro import KokoroTTSArgs

"""
# if not espeak-ng, brew update
MODELS_DIR=./models
brew install espeak-ng
huggingface-cli download hexgrad/Kokoro-82M --quiet --local-dir $MODELS_DIR/Kokoro82M
touch $MODELS_DIR/__init__.py
"""
try:
    sys.path.insert(1, os.path.join(MODELS_DIR, "Kokoro82M"))
    import torch
    from models.Kokoro82M.kokoro import generate
    from models.Kokoro82M.models import build_model
except ModuleNotFoundError as e:
    logging.error("In order to use kokoro-tts, you need to `pip install achatbot[tts_kokoro]`.")
    raise Exception(f"Missing module: {e}")


class KokoroTorchTTS(BaseTTS, ITts):
    """
    https://github.com/ai-bot-pro/achatbot/pull/105
    """

    TAG = "tts_kokoro"

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**KokoroTTSArgs().__dict__, **kwargs}

    def __init__(self, **args) -> None:
        self.args = KokoroTTSArgs(**args)
        self.args.device = self.args.device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        logging.debug(f"{KokoroTorchTTS.TAG} args: {self.args}")

        self.model = build_model(self.args.ckpt_path, self.args.device)
        total_params = 0
        for key, model in self.model.items():
            logging.debug(f"{key} Model: {model}")
            params = sum(p.numel() for p in model.parameters())
            total_params += params
            model_million_params = params / 1e6
            logging.debug(f"{key} Model has {model_million_params:.3f} million parameters")

        model_million_params = total_params / 1e6
        logging.debug(f"Model total has {model_million_params:.3f} million parameters")

        self.voices = None
        self.voice_stats = None

        self.load_voices(is_reload=True)
        self.set_voice(self.args.voice)

    def load_voices(self, is_reload=False):
        if self.voices and not is_reload:
            return

        voices_pt_file_list = glob.glob(os.path.join(self.args.voices_stats_dir, "*.pt"))
        # logging.debug(voices_pt_file_list)
        self.voices = {}
        for voice_file in voices_pt_file_list:
            voice_stats = torch.load(voice_file, weights_only=True)
            voice = os.path.splitext(os.path.basename(voice_file))[0]
            self.voices[voice] = voice_stats

    def get_voices(self) -> list[str]:
        return list(self.voices.keys())

    def set_voice(self, voice: str):
        if voice in self.get_voices():
            self.voice_stats = self.voices[voice]
            self.args.voice = voice

    def get_stream_info(self) -> dict:
        return {
            # "format": PYAUDIO_PAINT16,
            "format": PYAUDIO_PAFLOAT32,
            "channels": 1,
            "rate": 24000,  # target_sample_rate
            "sample_width": 2,
            # "np_dtype": np.int16,
            "np_dtype": np.float32,
        }

    async def _inference(
        self, session: Session, text: str, **kwargs
    ) -> AsyncGenerator[bytes, None]:
        audio_samples, _ = generate(
            self.model, text, self.voice_stats, lang=self.args.voice[0], speed=self.args.speed
        )

        yield np.frombuffer(audio_samples, dtype=np.float32).tobytes()
