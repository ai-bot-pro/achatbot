import os
from typing import AsyncGenerator
import asyncio
import re

import librosa

from src.core.llm.transformers.manual_vision_speech_gemma3n import TransformersManualAudioGemma3nLM
from src.common.utils.audio_utils import bytes2NpArrayWith16
from src.common.session import Session
from src.modules.speech.asr.base import ASRBase


class Gemma3nAsr(ASRBase):
    TAG = "gemma3n_asr"

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return kwargs

    def __init__(self, **args) -> None:
        args["init_chat_prompt"] = "You are a speech recognition model."
        self.model = TransformersManualAudioGemma3nLM(**args)
        self.args = args
        self.default_prompt = []

    def set_audio_data(self, audio_data):
        audio_nparr = None
        if isinstance(audio_data, (bytes, bytearray)):
            audio_nparr = bytes2NpArrayWith16(audio_data)
        if isinstance(audio_data, str):  # path
            audio_nparr, _ = librosa.load(audio_data, sr=16000, mono=True)
        self.default_prompt = (
            [
                {
                    "type": "text",
                    "text": "Based on the attached audio, generate a comprehensive text transcription of the spoken content.",
                },
                {"type": "audio", "audio": self.asr_audio},
            ]
            if audio_nparr
            else []
        )

    async def transcribe_stream(self, session: Session) -> AsyncGenerator[str, None]:
        session.ctx.state["prompt"] = session.ctx.state.get("prompt", self.default_prompt.copy())
        transcription = self.model.generate(session)
        for text in transcription:
            yield text

    async def transcribe(self, session: Session) -> dict:
        res = ""
        async for text in self.transcribe_stream(session):
            res += text

        res = {
            "language": self.args.get("language", "zh"),
            "language_probability": None,
            "text": res,
            "words": [],
        }
        return res
