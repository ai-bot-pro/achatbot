import logging
import os

import torch
import torchaudio

from src.common.utils.audio_utils import (
    bytes2NpArrayWith16,
    bytes2TorchTensorWith16,
)
from src.common.session import Session
from src.common.types import (
    SileroVADArgs,
    RATE, CHUNK, INT16_MAX_ABS_VALUE,
)
from .base import BaseVAD


class SileroVAD(BaseVAD):
    TAG = "silero_vad"
    map_rate_num_samples = {
        16000: 512,
        8000: 256,
    }

    def __init__(self, **args: SileroVADArgs) -> None:
        self.args = SileroVADArgs(**args)
        torch.set_num_threads(os.cpu_count())
        # torchaudio.set_audio_backend("soundfile")
        self.model, _ = torch.hub.load(
            repo_or_dir=self.args.repo_or_dir,
            model=self.args.model,
            source=self.args.source,
            force_reload=self.args.force_reload,
            onnx=self.args.onnx,
            verbose=self.args.verbose,
        )
        model_million_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        logging.debug(f"{self.TAG} have {model_million_params}M parameters")
        logging.debug(self.model)

    async def detect(self, session: Session):
        audio_chunk = bytes2NpArrayWith16(self.audio_buffer)
        if len(audio_chunk) != self.map_rate_num_samples[16000]:
            if self.args.is_pad_tensor is False:
                logging.debug(
                    f"len(audio_chunk):{len(audio_chunk)} dont't pad to {self.map_rate_num_samples[16000]} return False")
                return False
            logging.debug(
                f"len(audio_chunk):{len(audio_chunk)} pad to {self.map_rate_num_samples[16000]} ")
            audio_chunk = torch.nn.functional.pad(
                torch.from_numpy(audio_chunk),
                (0, self.map_rate_num_samples[16000] - len(audio_chunk)),
                "constant",
                0,
            ).numpy()
        vad_prob = self.model(torch.from_numpy(audio_chunk), RATE).item()
        is_silero_speech_active = vad_prob > (1 - self.args.silero_sensitivity)
        return is_silero_speech_active

    def close(self):
        self.model.reset_states()
