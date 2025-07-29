import base64
import logging
from typing import AsyncGenerator

from dotenv import load_dotenv
import numpy as np
import torch


from src.common.utils.helper import get_device
from src.common.random import set_all_random_seed
from src.common.interface import ITts
from src.common.session import Session
from src.common.types import PYAUDIO_PAFLOAT32, PYAUDIO_PAINT16
from src.types.speech.tts.higgs import HiggsTTSArgs
from src.types.llm.transformers import TransformersSpeechLMArgs
from .base import BaseTTS

from src.core.llm.transformers.manual_speech_higgs import (
    TransformersManualSpeechHiggs,
    ChatMLSample,
    Message,
    AudioContent,
    revert_delay_pattern,
)

load_dotenv(override=True)


def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a content from a local file to base64 format."""
    # Read the audio file as binary and encode it directly to Base64
    with open(file_path, "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
    return audio_base64


class HiggsTTS(BaseTTS, ITts):
    r"""
    https://github.com/ai-bot-pro/achatbot/pull/177
    """

    TAG = "tts_higgs"

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**HiggsTTSArgs().__dict__, **kwargs}

    def __init__(self, **args) -> None:
        self.args = HiggsTTSArgs(**args)
        self.args.device = self.args.device or get_device()
        logging.info(f"{HiggsTTS.TAG} args: {self.args}")

        self.lm_args = TransformersSpeechLMArgs(**self.args.lm_args)
        lm_args_dict = self.lm_args.__dict__
        lm_args_dict["lm_device"] = self.args.device
        lm_args_dict["audio_tokenizer_path"] = self.args.audio_tokenizer_path
        self.lm_model = TransformersManualSpeechHiggs(**self.lm_args.__dict__)
        self.lm_tokenizer = self.lm_model.serve_engine.tokenizer
        self.audio_tokenizer = self.lm_model.serve_engine.audio_tokenizer
        self.audio_stream_bos_id = self.lm_model.serve_engine.model.config.audio_stream_bos_id
        self.audio_stream_eos_id = self.lm_model.serve_engine.model.config.audio_stream_eos_id

        self.voices = {}
        self.set_voice(self.args.ref_audio_path, ref_text=self.args.ref_text, ref_speaker="default")

    def get_stream_info(self) -> dict:
        return {
            # "format": PYAUDIO_PAINT16,
            "format": PYAUDIO_PAFLOAT32,
            "channels": 1,
            "rate": self.audio_tokenizer.sample_rate,  # target_sample_rate
            "sample_width": 2,
            # "np_dtype": np.int16,
            "np_dtype": np.float32,
        }

    def set_voice(self, ref_file: str, **kwargs):
        ref_text = kwargs["ref_text"] if "ref_text" in kwargs else self.args.ref_text
        ref_speaker = kwargs["ref_speaker"] if "ref_speaker" in kwargs else ref_file
        reference_audio = encode_base64_content_from_file(ref_file)
        self.voices[ref_speaker] = [
            Message(
                role=self.lm_args.user_role,
                content=ref_text,
            ),
            Message(
                role=self.lm_args.assistant_role,
                content=AudioContent(raw_audio=reference_audio, audio_url="placeholder"),
            ),
        ]

    def get_voices(self):
        return list(self.voices.keys())

    async def _inference(
        self, session: Session, text: str, **kwargs
    ) -> AsyncGenerator[bytes, None]:
        seed = kwargs.get("seed", self.lm_args.lm_gen_seed)
        set_all_random_seed(seed)

        ref_speaker = kwargs["ref_speaker"] if "ref_speaker" in kwargs else "default"
        ref_voice_messages = self.voices.get(ref_speaker)
        if ref_voice_messages is None:
            logging.warning(f"Voice {ref_speaker} not found, use random speaker.")
            ref_voice_messages = []

        messages = [
            Message(
                role=self.lm_args.init_chat_role,
                content=self.lm_args.init_chat_prompt or self.lm_model.DEFAULT_SYS_PROMPT,
            )
        ]
        messages.extend(ref_voice_messages)
        messages.append(
            Message(
                role="user",
                content=text,
            )
        )
        session.ctx.state["messages"] = messages
        streamer = self.lm_model.async_generate(session, **kwargs)

        audio_tokens = []
        audio_tensor = None
        CHUNK_SIZE = kwargs.get("chunk_size", self.args.chunk_size)
        seq_len = 0

        with torch.inference_mode():
            async for delta in streamer:
                if delta["audio_vq_tokens"] is None:
                    continue

                if torch.all(delta["audio_vq_tokens"] == self.audio_stream_eos_id):
                    break

                audio_tokens.append(delta["audio_vq_tokens"][:, None])
                audio_tensor = torch.cat(audio_tokens, dim=-1)

                if torch.all(delta["audio_vq_tokens"] != self.audio_stream_bos_id):
                    seq_len += 1
                if seq_len > 0 and seq_len % CHUNK_SIZE == 0:
                    vq_code = (
                        revert_delay_pattern(audio_tensor, start_idx=seq_len - CHUNK_SIZE + 1)
                        .clip(0, 1023)
                        .to(self.lm_args.lm_device)
                    )

                    audio_np = self.lm_model.serve_engine.audio_tokenizer.decode(
                        vq_code.unsqueeze(0)
                    )[0, 0]
                    # audio_np = (audio_np * 32767).astype(np.int16)
                    yield audio_np.tobytes()

            if seq_len > 0 and seq_len % CHUNK_SIZE != 0 and audio_tensor is not None:
                vq_code = (
                    revert_delay_pattern(audio_tensor, start_idx=seq_len - seq_len % CHUNK_SIZE + 1)
                    .clip(0, 1023)
                    .to(self.lm_args.lm_device)
                )

                audio_np = self.lm_model.serve_engine.audio_tokenizer.decode(vq_code.unsqueeze(0))[
                    0, 0
                ]
                # audio_np = (audio_np * 32767).astype(np.int16)
                yield audio_np.tobytes()

        torch.cuda.empty_cache()
