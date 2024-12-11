from pathlib import Path
from typing import AsyncGenerator
import logging

from src.common.utils.wav import save_audio_to_file
from src.common.session import Session
from src.common.types import RECORDS_DIR
from src.types.speech.asr.whisper import WhisperGroqASRArgs
from src.modules.speech.asr.base import ASRBase


class WhisperGroqAsr(ASRBase):
    TAG = "whisper_groq_asr"

    def __init__(self, **args) -> None:
        from groq import Groq

        self.args = WhisperGroqASRArgs(**args)
        self.asr_audio = None
        self.client = Groq()

    def set_audio_data(self, audio_data):
        if isinstance(audio_data, (bytes, bytearray)):
            self.asr_audio = bytes(audio_data)
        if isinstance(audio_data, str):
            self.asr_audio = Path(audio_data)

    async def transcribe_stream(self, session: Session) -> AsyncGenerator[str, None]:
        if isinstance(self.asr_audio, bytes):
            file_path = await save_audio_to_file(self.asr_audio, "tmp.wav", audio_dir=RECORDS_DIR)
            self.asr_audio = Path(file_path)
        transcription = self.client.audio.transcriptions.create(
            file=self.asr_audio,
            model=self.args.model_name_or_path,
            prompt=self.args.prompt,  # Optional
            response_format="text",  # Optional
            language=self.args.language,  # Optional
            temperature=self.args.temperature,  # Optional
            timeout=self.args.timeout_s,  # Optional
        )
        yield transcription

    async def transcribe(self, session: Session) -> dict:
        if isinstance(self.asr_audio, bytes):
            file_path = await save_audio_to_file(self.asr_audio, "tmp.wav", audio_dir=RECORDS_DIR)
            self.asr_audio = Path(file_path)

        transcription = self.client.audio.transcriptions.create(
            file=self.asr_audio,
            model=self.args.model_name_or_path,
            prompt=self.args.prompt,  # Optional
            response_format="verbose_json",  # Optional
            language=self.args.language,  # Optional
            temperature=self.args.temperature,  # Optional
            timeout=self.args.timeout_s,  # Optional
        )
        res = {
            "language": self.args.language,
            "language_probability": transcription.language,
            "text": transcription.text,
            "segments": transcription.segments,
            "words": [],
        }
        return res
