import logging
import os
from pathlib import Path
import sys
from typing import AsyncGenerator

import hydra
from hydra import compose, initialize
from hydra.utils import instantiate
import numpy as np
import torch
import torchaudio

try:
    cur_dir = os.path.dirname(__file__)
    if bool(os.getenv("ACHATBOT_PKG", "")):
        sys.path.insert(1, os.path.join(cur_dir, "../../../Zonos"))
    else:
        sys.path.insert(1, os.path.join(cur_dir, "../../../../deps/Zonos"))
except ModuleNotFoundError as e:
    logging.error(
        "In order to use zonos-tts, you need to `pip install achatbot[tts_zonos]`."
    )
    raise Exception(f"Missing module: {e}")

from src.common.types import PYAUDIO_PAFLOAT32, PYAUDIO_PAINT16
from src.common.utils.helper import file_md5_hash, get_device, print_model_params
from src.common.interface import ITts
from src.common.session import Session
from src.types.speech.tts.fish_speech import FishSpeechTTSArgs
from .base import BaseTTS


class FishSpeechTTS(BaseTTS, ITts):
    TAG = "tts_fishspeech"

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**FishSpeechTTSArgs().__dict__, **kwargs}

    def __init__(self, **args) -> None:
        self.args = FishSpeechTTSArgs(**args)
        self.args.device = self.args.device or get_device()
