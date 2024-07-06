
from src.common.session import Session
from .base import PyAudioRecorder


class VADRecorder(PyAudioRecorder):
    TAG = "vad_recorder"

    def record_audio(self, session: Session) -> list[bytes]:
        pass


class WakeWordsVADRecorder(PyAudioRecorder):
    TAG = ""

    def record_audio(self, session: Session) -> list[bytes]:
        pass
