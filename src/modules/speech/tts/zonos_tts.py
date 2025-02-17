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
    logging.error("In order to use zonos-tts, you need to `pip install achatbot[tts_zonos]`.")
    raise Exception(f"Missing module: {e}")

from src.common.types import PYAUDIO_PAFLOAT32, PYAUDIO_PAINT16
from src.common.utils.helper import file_md5_hash, get_device, print_model_params
from src.common.interface import ITts
from src.common.session import Session
from src.types.speech.tts.zonos import ZonosTTSArgs
from .base import BaseTTS


class ZonosSpeechTTS(BaseTTS, ITts):
    TAG = "tts_zonos"

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**ZonosTTSArgs().__dict__, **kwargs}

    def __init__(self, **args) -> None:
        self.args = ZonosTTSArgs(**args)
        self.args.device = self.args.device or get_device()

    def get_stream_info(self) -> dict:
        return {
            # "format": PYAUDIO_PAINT16,
            "format": PYAUDIO_PAFLOAT32,
            "channels": 1,
            # "rate": self.gan_model.spec_transform.sample_rate,
            # https://huggingface.co/descript/dac_44khz/blob/main/config.json
            "rate": 44100,
            "sample_width": 2,
            # "np_dtype": np.int16,
            "np_dtype": np.float32,
        }
