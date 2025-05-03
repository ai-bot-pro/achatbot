import os
from typing import AsyncGenerator
import asyncio
import re

import librosa

from src.core.llm.transformers.manual_voice_kimi import TransformersManualAudioKimiLLM
from src.common.utils.audio_utils import bytes2TorchTensorWith16
from src.common.session import Session
from src.modules.speech.asr.base import ASRBase


class KimiAsr(ASRBase):
    TAG = "kimi_asr"

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return kwargs

    def __init__(self, **args) -> None:
        self.model = TransformersManualAudioKimiLLM(**args)
        self.args = args

    def set_audio_data(self, audio_data):
        if isinstance(audio_data, (bytes, bytearray)):
            self.asr_audio = bytes2TorchTensorWith16(audio_data)
        if isinstance(audio_data, str):  # path
            self.asr_audio = audio_data

    async def transcribe_stream(self, session: Session) -> AsyncGenerator[str, None]:
        prompt = [
            {
                "role": "user",
                "message_type": "text",
                "content": "Please transcribe the following audio:",
            },
            {
                "role": "user",
                "message_type": "audio",
                "content": self.asr_audio,
            },
        ]
        session.ctx.state["prompt"] = session.ctx.state.get("prompt", prompt)
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
