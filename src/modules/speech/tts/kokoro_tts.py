from typing import AsyncGenerator, Coroutine
from common.session import Session
from src.common.interface import ITts
from src.modules.speech.tts.base import BaseTTS


class KokoroOnnxTTS(BaseTTS, ITts):
    TAG = "tts_kokoro_onnx"

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**KokoroOnnxTTSArgs().__dict__, **kwargs}

    def __init__(self, **args) -> None:
        self.args = KokoroOnnxTTSArgs(**args)

    def get_voices(self) -> list[str]:
        return ["kokoro"]

    def set_voice(self, voice: str):
        return super().set_voice(voice)

    def _inference(self, session: Session, text: str) -> AsyncGenerator[bytes, None]:
        return super()._inference(session, text)
    
