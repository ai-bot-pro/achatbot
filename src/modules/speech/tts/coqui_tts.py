import logging
from typing import AsyncGenerator
import time
import os
import random

import torch
import numpy as np

from src.common.device_cuda import CUDAInfo
from src.common.interface import ITts
from src.common.session import Session
from src.common.types import PYAUDIO_PAFLOAT32, CoquiTTSArgs
from src.common.utils.audio_utils import postprocess_tts_wave
from .base import BaseTTS


class CoquiTTS(BaseTTS, ITts):
    TAG = "tts_coqui"

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**CoquiTTSArgs().__dict__, **kwargs}

    def __init__(self, **args) -> None:
        from TTS.api import TTS
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        from TTS.tts.layers.xtts.xtts_manager import SpeakerManager

        self.args = CoquiTTSArgs(**args)
        logging.debug(f"{self.TAG} args: {self.args}")
        info = CUDAInfo()
        config = XttsConfig()
        config.load_json(self.args.conf_file)
        model = Xtts.init_from_config(config)
        logging.debug("Loading model...")
        model.load_checkpoint(
            config,
            checkpoint_dir=self.args.model_path,
            use_deepspeed=self.args.tts_use_deepspeed,
        )
        model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
        logging.debug(f"{model_million_params}M parameters")

        if info.is_cuda:
            model.cuda()
        self.model = model
        self.config = config
        speaker_file_path = os.path.join(self.args.model_path, "speakers_xtts.pth")
        self.speaker_manager = SpeakerManager(speaker_file_path)

        self.set_reference_audio(self.args.reference_audio_path)

    def set_reference_audio(self, reference_audio_path: str):
        logging.debug("Computing speaker latents...")
        if not os.path.exists(reference_audio_path):
            voices = self.get_voices()
            voice = random.choice(voices)
            self.gpt_cond_latent, self.speaker_embedding = self.speaker_manager.speakers[
                voice
            ].values()
            logging.debug(
                f"reference_audio_path {reference_audio_path} don't exist; use speaker {voice}"
            )
            return

        self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(
            audio_path=[reference_audio_path], gpt_cond_len=30, max_ref_length=60
        )

    def get_stream_info(self) -> dict:
        return {
            "format": PYAUDIO_PAFLOAT32,
            "channels": 1,
            "rate": self.config.audio.output_sample_rate,
            "sample_width": 4,
            "np_dtype": np.float32,
        }

    async def _inference(
        self, session: Session, text: str, **kwargs
    ) -> AsyncGenerator[bytes, None]:
        if self.args.tts_stream is False:
            logging.debug("Inference...")
            out = self.model.inference(
                text,
                self.args.tts_language,
                self.gpt_cond_latent,
                self.speaker_embedding,
                temperature=self.args.tts_temperature,
                top_p=self.args.tts_top_p,
                length_penalty=self.args.tts_length_penalty,
                repetition_penalty=self.args.tts_repetition_penalty,
                speed=self.args.tts_speed,
                num_beams=self.args.tts_num_beams,
            )

            tensor_wave = torch.tensor(out["wav"]).unsqueeze(0).cpu()
            logging.debug(
                f"{self.TAG} inference out tensor {torch.tensor(out['wav']).shape}, tensor_wave: {tensor_wave.shape}"
            )
            # torchaudio.save("records/tts_coqui_infer_zh_test.wav", tensor_wave, 24000)

            res = postprocess_tts_wave(tensor_wave)
            yield res

        else:
            logging.debug("Inference streaming...")
            time_start = time.time()
            chunks = self.model.inference_stream(
                text,
                self.args.tts_language,
                self.gpt_cond_latent,
                self.speaker_embedding,
                temperature=self.args.tts_temperature,
                top_p=self.args.tts_top_p,
                length_penalty=self.args.tts_length_penalty,
                repetition_penalty=self.args.tts_repetition_penalty,
                speed=self.args.tts_speed,
                stream_chunk_size=self.args.tts_stream_chunk_size,
                overlap_wav_len=self.args.tts_overlap_wav_len,
                enable_text_splitting=self.args.tts_enable_text_splitting,
            )
            seconds_to_first_chunk = 0.0
            full_generated_seconds = 0.0
            raw_inference_start = 0.0
            first_chunk_length_seconds = 0.0
            for i, chunk in enumerate(chunks):
                logging.debug(f"Received chunk {i} of audio length {chunk.shape[-1]}")
                chunk = postprocess_tts_wave(chunk)
                yield chunk
                # 4 bytes per sample, 24000 Hz
                chunk_duration = len(chunk) / (4 * self.config.audio.output_sample_rate)
                full_generated_seconds += chunk_duration
                if i == 0:
                    first_chunk_length_seconds = chunk_duration
                    raw_inference_start = time.time()
                    seconds_to_first_chunk = raw_inference_start - time_start
            self._print_synthesized_info(
                time_start,
                full_generated_seconds,
                first_chunk_length_seconds,
                seconds_to_first_chunk,
            )

    def _print_synthesized_info(
        self, time_start, full_generated_seconds, first_chunk_length_seconds, seconds_to_first_chunk
    ):
        time_end = time.time()
        seconds = time_end - time_start
        if full_generated_seconds > 0 and (full_generated_seconds - first_chunk_length_seconds) > 0:
            realtime_factor = seconds / full_generated_seconds
            raw_inference_time = seconds - seconds_to_first_chunk
            raw_inference_factor = raw_inference_time / (
                full_generated_seconds - first_chunk_length_seconds
            )

            logging.debug(
                f"XTTS synthesized {full_generated_seconds:.2f}s"
                f" audio in {seconds:.2f}s"
                f" realtime factor: {realtime_factor:.2f}x"
            )
            logging.debug(
                f"seconds to first chunk: {seconds_to_first_chunk:.2f}s"
                f" raw_inference_factor: {raw_inference_factor:.2f}x"
            )

    def get_voices(self):
        voices_list = []
        for speaker_name in self.speaker_manager.name_to_id:
            voices_list.append(speaker_name)
        return voices_list
