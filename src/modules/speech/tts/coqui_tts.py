import logging
import asyncio
from typing import Iterator
import time

import torch
import numpy as np
import torchaudio

from src.common.device_cuda import CUDAInfo
from src.common.interface import ITts
from src.common.factory import EngineClass
from src.common.session import Session
from src.common.types import CoquiTTSArgs
from src.common.utils.audio_utils import postprocess_tts_wave


class CoquiTTS(ITts, EngineClass):
    TAG = "tts_coqui"

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**CoquiTTSArgs().__dict__, **kwargs}

    def __init__(self, **args) -> None:
        from TTS.api import TTS
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        self.args = CoquiTTSArgs(**args)
        info = CUDAInfo()
        logging.debug("inference Loading model...")
        config = XttsConfig()
        config.load_json(self.args.conf_file)
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=self.args.model_path,
                              use_deepspeed=info.is_cuda)
        model_million_params = sum(p.numel() for p in model.parameters())/1e6
        logging.debug(f"{model_million_params}M parameters")

        if info.is_cuda:
            model.cuda()
        self.model = model
        self.config = config

        asyncio.run(self.set_reference_audio(self.args.reference_audio_path))

    async def set_reference_audio(self, reference_audio_path: str):
        logging.debug("Computing speaker latents...")
        self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(
            audio_path=[reference_audio_path],
            gpt_cond_len=30, max_ref_length=60)

    def synthesize(self, session: Session) -> Iterator[bytes]:
        if "tts_text_iter" in session.ctx.state:
            for text in session.ctx.state["tts_text_iter"]:
                for chunk in self._inference(session, text):
                    yield chunk
                yield self._get_end_silence_chunk(session, text)
        elif "tts_text" in session.ctx.state:
            text = session.ctx.state["tts_text"]
            for chunk in self._inference(session, text):
                yield chunk
            yield self._get_end_silence_chunk(session, text)

    def _inference(self, session: Session, text: str) -> Iterator[bytes]:
        if session.ctx.tts_stream is False:
            logging.debug("Inference...")
            out = self.model.inference(
                text,
                session.ctx.tts_language,
                self.gpt_cond_latent,
                self.speaker_embedding,
                temperature=session.ctx.tts_temperature,
                top_p=session.ctx.tts_top_p,
                length_penalty=session.ctx.tts_length_penalty,
                repetition_penalty=session.ctx.tts_repetition_penalty,
                speed=session.ctx.tts_speed,
                num_beams=session.ctx.tts_num_beams,
            )
            tensor_wave = torch.tensor(out["wav"]).unsqueeze(0).cpu()
            # torchaudio.save("records/tts_coqui_infer_zh_test.wav", tensor_wave, 24000)

            res = postprocess_tts_wave(tensor_wave)
            yield res

        else:
            logging.debug("Inference streaming...")
            time_start = time.time()
            chunks = self.model.inference_stream(
                text,
                session.ctx.tts_language,
                self.gpt_cond_latent,
                self.speaker_embedding,
                temperature=session.ctx.tts_temperature,
                top_p=session.ctx.tts_top_p,
                length_penalty=session.ctx.tts_length_penalty,
                repetition_penalty=session.ctx.tts_repetition_penalty,
                speed=session.ctx.tts_speed,
                stream_chunk_size=session.ctx.tts_stream_chunk_size,
                overlap_wav_len=session.ctx.tts_overlap_wav_len,
                enable_text_splitting=session.ctx.tts_enable_text_splitting
            )
            seconds_to_first_chunk = 0.0
            full_generated_seconds = 0.0
            raw_inference_start = 0.0
            first_chunk_length_seconds = 0.0
            for i, chunk in enumerate(chunks):
                logging.debug(
                    f"Received chunk {i} of audio length {chunk.shape[-1]}")
                chunk = postprocess_tts_wave(chunk)
                yield chunk
                # 4 bytes per sample, 24000 Hz
                chunk_duration = len(chunk) / \
                    (4 * self.config.audio.output_sample_rate)
                full_generated_seconds += chunk_duration
                if i == 0:
                    first_chunk_length_seconds = chunk_duration
                    raw_inference_start = time.time()
                    seconds_to_first_chunk = raw_inference_start - time_start
            self._print_synthesized_info(time_start, full_generated_seconds,
                                         first_chunk_length_seconds,
                                         seconds_to_first_chunk)

    def _print_synthesized_info(self, time_start, full_generated_seconds,
                                first_chunk_length_seconds, seconds_to_first_chunk):
        time_end = time.time()
        seconds = time_end - time_start
        if full_generated_seconds > 0 \
                and (full_generated_seconds - first_chunk_length_seconds) > 0:
            realtime_factor = seconds / full_generated_seconds
            raw_inference_time = seconds - seconds_to_first_chunk
            raw_inference_factor = raw_inference_time / \
                (full_generated_seconds - first_chunk_length_seconds)

            logging.debug(
                f"XTTS synthesized {full_generated_seconds:.2f}s"
                f" audio in {seconds:.2f}s"
                f" realtime factor: {realtime_factor:.2f}x"
            )
            logging.debug(
                f"seconds to first chunk: {seconds_to_first_chunk:.2f}s"
                f" raw_inference_factor: {raw_inference_factor:.2f}x"
            )

    def _get_end_silence_chunk(self, session: Session, text: str) -> bytes:
        # Send silent audio
        sample_rate = self.config.audio.sample_rate

        end_sentence_delimeters = ".。!！?？…。¡¿"
        mid_sentence_delimeters = ";；:：,，\n（()）【[]】「{}」-“”„\"—/|《》"

        if text[-1] in end_sentence_delimeters:
            silence_duration = session.ctx.tts_sentence_silence_duration
        elif text[-1] in mid_sentence_delimeters:
            silence_duration = session.ctx.tts_comma_silence_duration
        else:
            silence_duration = session.ctx.tts_default_silence_duration

        silent_samples = int(sample_rate * silence_duration)
        silent_chunk = np.zeros(silent_samples, dtype=np.float32)
        logging.debug(f"add silent_chunk {silent_chunk.shape}")
        return silent_chunk.tobytes()
