from typing import Iterator

from src.common.factory import EngineClass
from src.common.session import Session


class TTSVoice:
    def __init__(self, name, id):
        self.name = name
        self.id = id

    def __repr__(self):
        return f"<TTSVoice(name={self.name} id={self.id})>"


class BaseTTS(EngineClass):
    def synthesize(self, session: Session) -> Iterator[bytes]:
        if "tts_text_iter" in session.ctx.state:
            for text in session.ctx.state["tts_text_iter"]:
                for chunk in self._inference(session, text):
                    yield chunk
                silence_chunk = self._get_end_silence_chunk(session, text)
                if silence_chunk:
                    yield silence_chunk
        elif "tts_text" in session.ctx.state:
            text = session.ctx.state["tts_text"]
            for chunk in self._inference(session, text):
                yield chunk
            silence_chunk = self._get_end_silence_chunk(session, text)
            if silence_chunk:
                yield silence_chunk

    def _inference(self, session: Session, text: str) -> Iterator[bytes]:
        raise NotImplementedError(
            "The _inference method must be implemented by the derived subclass.")

    def _get_end_silence_chunk(self, session: Session, text: str) -> bytes:
        b''
