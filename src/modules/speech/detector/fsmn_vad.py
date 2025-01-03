import logging
import math

import numpy as np

from src.common.utils.audio_utils import bytes2NpArrayWith16
from src.common.types import (
    VAD_CHECK_PER_FRAMES,
    VAD_CHECK_ALL_FRAMES,
    FSMNVADArgs,
)
from src.common.session import Session
from .base import BaseVAD


class FSMNVAD(BaseVAD):
    TAG = "fsmn_vad"
    map_rate_num_samples = {
        16000: 1024,  # rate: frame_length 1024 = 16000*64 / 1000
    }

    def __init__(self, **args) -> None:
        from funasr import AutoModel

        self.args = FSMNVADArgs(**args)
        self.model = AutoModel(model=self.args.model, model_revision=self.args.model_version)
        logging.debug(self.model)
        self.audio_buffer = None
        self.cache = {}

    def get_sample_info(self):
        return self.args.sample_rate, self.map_rate_num_samples[self.args.sample_rate]

    def set_audio_data(self, audio_data):
        super().set_audio_data(audio_data)
        if isinstance(self.audio_buffer, (bytes, bytearray)):
            self.audio_buff = bytes2NpArrayWith16(self.audio_buffer)

    async def detect(self, session: Session):
        if self.args.check_frames_mode not in [VAD_CHECK_ALL_FRAMES, VAD_CHECK_PER_FRAMES]:
            return False

        speech_frames = 0

        # Number of audio frames per millisecond
        # frame_length = int(
        #    self.args.sample_rate * self.args.frame_duration_ms / 1000
        # )  # 64 ms (16000) -> 1024
        ## just process channels:1 sample_width:2 audio
        # num_frames = math.ceil(len(self.audio_buffer) / frame_length)

        # window_size_samples = frame_length
        frame_length = self.map_rate_num_samples[self.args.sample_rate]
        num_frames = math.ceil(len(self.audio_buff) / frame_length)
        logging.debug(
            f"{self.TAG} Speech detected audio_len:{len(self.audio_buffer)} frame_len:{frame_length} num_frames {num_frames}"
        )

        for i in range(num_frames):
            chunk = self.audio_buff[i * frame_length : (i + 1) * frame_length]
            res = await self.detect_chunk(chunk, session)
            if res is True:
                speech_frames += 1
                if self.args.check_frames_mode == VAD_CHECK_PER_FRAMES:
                    logging.debug(
                        f"{self.TAG} Speech detected in frame {i + 1}" f" of {num_frames}"
                    )
                    # await save_audio_to_file(frame, "fsmn_vad_frame.wav")
                    return True

        if self.args.check_frames_mode == VAD_CHECK_ALL_FRAMES:
            if speech_frames == num_frames:
                logging.debug(
                    f"{self.TAG} Speech detected in {speech_frames} of " f"{num_frames} frames"
                )
            else:
                logging.debug(f"{self.TAG} Speech not detected in all {num_frames} frames")
            return speech_frames == num_frames

        logging.debug(f"{self.TAG} Speech not detected in any of {num_frames} frames")
        return False

    async def detect_chunk(self, chunk, session: Session):
        audio_chunk = self.process_audio_buffer(chunk)
        chunk_size = len(audio_chunk) / 16  # frame_duration_ms 1024*1000/16000 = 64 ms
        res = self.model.generate(
            input=audio_chunk,
            cache=self.cache,
            is_final=False,
            chunk_size=chunk_size,
            disable_pbar=True,
        )
        logging.debug(f"{res}")
        if len(res[0]["value"]):
            if res[0]["value"][0][0] != -1:
                return True
        return False

    def process_audio_buffer(self, buffer):
        audio_chunk = buffer
        if isinstance(buffer, (bytes, bytearray)):
            audio_chunk = bytes2NpArrayWith16(buffer)
        if self.map_rate_num_samples[self.args.sample_rate] > len(audio_chunk):
            logging.debug(
                f"len(audio_chunk):{len(audio_chunk)} pad to {self.map_rate_num_samples[self.args.sample_rate]} "
            )
            audio_chunk = np.pad(
                audio_chunk,
                (0, self.map_rate_num_samples[self.args.sample_rate] - len(audio_chunk)),
                "constant",
                constant_values=(0, 0),
            )
        return audio_chunk
