import logging
import os
import math

import torch
import torchaudio

from src.common.utils.audio_utils import (
    bytes2NpArrayWith16,
    bytes2TorchTensorWith16,
)
from src.common.session import Session
from src.common.types import (
    VAD_CHECK_PER_FRAMES,
    VAD_CHECK_ALL_FRAMES,
    SileroVADArgs,
    RATE,
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
        torch.set_num_threads(1)
        # torchaudio.set_audio_backend("soundfile")
        self.model, utils = torch.hub.load(
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

        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         VADIterator,
         self.collect_chunks) = utils
        self.vad_iterator = VADIterator(self.model, sampling_rate=RATE)

    def set_audio_data(self, audio_data):
        super().set_audio_data(audio_data)
        if isinstance(self.audio_buffer, (bytes, bytearray)):
            self.audio_buff = torch.from_numpy(bytes2NpArrayWith16(self.audio_buffer))

    async def detect(self, session: Session):
        speech_frames = 0
        window_size_samples = self.map_rate_num_samples[16000]
        num_frames = math.ceil(len(self.audio_buff) / window_size_samples)
        for i in range(0, len(self.audio_buff), window_size_samples):
            chunk = self.audio_buff[i: i + window_size_samples]
            if await self.detect_chunk(chunk, session):
                speech_frames += 1
                if self.args.check_frames_mode == VAD_CHECK_PER_FRAMES:
                    logging.debug(f"{self.TAG} Speech detected in frame {i + 1}"
                                  f" of {num_frames}")
                    return True
        if self.args.check_frames_mode == VAD_CHECK_ALL_FRAMES:
            if speech_frames == num_frames:
                logging.debug(f"{self.TAG} Speech detected in {speech_frames} of "
                              f"{num_frames} frames")
            else:
                logging.debug(f"{self.TAG} Speech not detected in all {num_frames} frames")
            return speech_frames == num_frames

        logging.debug(f"{self.TAG} Speech not detected in any of {num_frames} frames")
        return False

    async def detect_chunk(self, chunk, session: Session):
        audio_chunk = chunk
        if isinstance(chunk, (bytes, bytearray)):
            audio_chunk = torch.from_numpy(bytes2NpArrayWith16(chunk))
        if len(audio_chunk) != self.map_rate_num_samples[16000]:
            if self.args.is_pad_tensor is False:
                logging.debug(
                    f"len(audio_chunk):{len(audio_chunk)} dont't pad to {self.map_rate_num_samples[16000]} return False")
                return False
            logging.debug(
                f"len(audio_chunk):{len(audio_chunk)} pad to {self.map_rate_num_samples[16000]} ")
            audio_chunk = torch.nn.functional.pad(
                audio_chunk,
                (0, self.map_rate_num_samples[16000] - len(audio_chunk)),
                "constant",
                0,
            )
        vad_prob = self.model(audio_chunk, RATE).item()
        is_silero_speech_active = vad_prob > (1 - self.args.silero_sensitivity)
        return is_silero_speech_active

    def get_speech_timestamps(self):
        speech_timestamps = self.get_speech_timestamps(
            self.audio_buff, self.model, sampling_rate=RATE)
        logging.debug(f"speech_timestamps:{speech_timestamps}")
        return speech_timestamps

    async def save_audio(self, saved_file_path):
        # merge all speech chunks to one audio
        self.save_audio(
            saved_file_path,
            self.collect_chunks(
                self.get_speech_timestamps(),
                self.audio_buff),
            sampling_rate=RATE
        )

    def vad_iterator(self):
        window_size_samples = self.map_rate_num_samples[RATE]
        for i in range(0, len(self.audio_buff), window_size_samples):
            audio_chunk = self.audio_buff[i: i + window_size_samples]
            if len(audio_chunk) < window_size_samples:
                if self.args.is_pad_tensor is False:
                    logging.debug(
                        f"len(audio_chunk):{len(audio_chunk)} dont't pad to {self.map_rate_num_samples[16000]} return False")
                    continue
                logging.debug(
                    f"len(audio_chunk):{len(audio_chunk)} pad to {self.map_rate_num_samples[16000]} ")
                audio_chunk = torch.nn.functional.pad(
                    audio_chunk,
                    (0, self.map_rate_num_samples[16000] - len(audio_chunk)),
                    "constant",
                    0,
                )
            speech_dict = self.vad_iterator(audio_chunk, return_seconds=True)
            if speech_dict:
                yield speech_dict

    def get_sample_info(self):
        return RATE, self.map_rate_num_samples[RATE]

    def close(self):
        self.model.reset_states()
