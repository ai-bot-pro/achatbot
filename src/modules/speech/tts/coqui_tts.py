import logging
import asyncio
from typing import Iterator

from src.common.device_cuda import CUDAInfo
from src.common.interface import ITts
from src.common.factory import EngineClass
from src.common.session import Session
from src.common.types import CoquiTTSArgs
from src.common.utils.audio_utils import npArray2bytes


class CoquiTTS(ITts, EngineClass):
    TAG = "tts_coqui"

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**(CoquiTTSArgs).__dict__, **kwargs}

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

        asyncio.run(self.set_reference_audio(self.args.reference_audio_path))

    async def set_reference_audio(self, reference_audio_path: str):
        logging.debug("Computing speaker latents...")
        self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(
            audio_path=[reference_audio_path])

    def inference(self, session: Session) -> Iterator[bytearray]:
        if "llm_text_iter" in session.ctx.state:
            yield self._inference_iter(session, session.ctx.state["llm_text_iter"])
        elif "llm_text" in session.ctx.state:
            yield self._inference(session, session.ctx.state["llm_text"])

    def _inference(self, session: Session, text: str) -> Iterator[bytearray]:
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
            session.ctx.state["tts_audio_data"] = out["wav"]
            yield npArray2bytes(out["wav"])

        else:
            logging.debug("Inference streaming...")
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
            )
            wav_chuncks = []
            for i, chunk in enumerate(chunks):
                logging.debug(
                    f"Received chunk {i} of audio length {chunk.shape[-1]}")
                wav_chuncks.append(chunk)
                yield npArray2bytes(out["wav"])

            session.ctx.state["tts_audio_data"] = bytearray(wav_chuncks)

    def _inference_iter(self, session: Session, text_iter: Iterator[str]) -> Iterator[bytearray]:
        for text in text_iter:
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
                session.ctx.state["tts_audio_data"] = out["wav"]
                yield npArray2bytes(out["wav"])

            else:
                logging.debug("Inference streaming...")
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
                )
                wav_chuncks = []
                for i, chunk in enumerate(chunks):
                    logging.debug(
                        f"Received chunk {i} of audio length {chunk.shape[-1]}")
                    wav_chuncks.append(chunk)
                    yield npArray2bytes(out["wav"])

                session.ctx.state["tts_audio_data"] = bytearray(wav_chuncks)
