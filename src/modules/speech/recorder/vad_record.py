
from src.common.session import Session
from .base import PyAudioRecorder


class VADRecorder(PyAudioRecorder):
    TAG = "vad_recorder"

    async def record_audio(self, session: Session) -> list[bytes]:
        pass


class WakeWordsVADRecorder(PyAudioRecorder):
    TAG = ""

    async def record_audio(self, session: Session) -> list[bytes]:
        pass
