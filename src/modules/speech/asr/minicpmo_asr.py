import os
from typing import AsyncGenerator
import asyncio
import re

from src.core.llm.transformers.manual_vision_voice_minicpmo import TransformersManualAudioMiniCPMO
from src.common.utils.audio_utils import bytes2NpArrayWith16
from src.common.session import Session
from src.modules.speech.asr.base import ASRBase


class MiniCPMoAsr(ASRBase):
    TAG = "minicpmo_asr"

    def __init__(self, **args) -> None:
        self.model = TransformersManualAudioMiniCPMO(**args)

    def set_audio_data(self, audio_data):
        if isinstance(audio_data, (bytes, bytearray)):
            self.asr_audio = bytes2NpArrayWith16(audio_data)

    async def transcribe_stream(self, session: Session) -> AsyncGenerator[str, None]:
        prompt = ["", self.asr_audio]
        session.ctx.state["prompt"] = session.ctx.state.get("prompt", prompt)
        transcription, _ = await asyncio.to_thread(self.model.generate, session)
        for item in transcription:
            clean_text = re.sub(r"<\|.*?\|>", "", item["text"])
            yield clean_text

    async def transcribe(self, session: Session) -> dict:
        res = {}
        return res
