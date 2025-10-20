import logging
import os
import threading
import time

import numpy as np

from .base import BaseVADAnalyzer
from src.common.types import (
    VADAnalyzerArgs,
)
from src.thirdparty.ten_vad import TenVad


class TenVADAnalyzer(BaseVADAnalyzer):
    TAG = "ten_vad_analyzer"

    def __init__(self, **args):
        super().__init__(**args)
        lib_path = os.getenv("TEN_VAD_LIB_PATH", None)
        self.ten_vad = TenVad(
            self.num_frames_required(), self._args.confidence, lib_path=lib_path
        )  # Create a TenVad instance

    def num_frames_required(self) -> int:
        hop_size = 256  # 16 ms per frame for 16K hz sample rate
        # hop_size = 160  # 10 ms per frame for 16K hz sample rate
        return hop_size

    def reset(self):
        super().reset()

    def process_audio_buffer(self, buffer):
        audio_chunk = buffer
        if isinstance(buffer, (bytes, bytearray)):
            audio_chunk = np.frombuffer(buffer, dtype=np.int16)
        if not isinstance(audio_chunk, np.ndarray):
            raise Exception(f"buffer type error, expect bytes or np.ndarray, got {type(buffer)}")

        assert len(audio_chunk) == self.num_frames_required()

        return audio_chunk

    def voice_confidence(self, buffer) -> float:
        try:
            audio_chunk = self.process_audio_buffer(buffer)
            vad_prob, _ = self.ten_vad.process(audio_chunk)
            return vad_prob
        except Exception as ex:
            # This comes from an empty audio array
            logging.exception(f"Error analyzing audio with TEN-VAD: {ex}")
            return 0
