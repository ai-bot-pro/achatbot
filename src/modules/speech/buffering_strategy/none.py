import time
import asyncio

from src.common.interface import IBuffering
from src.common.factory import EngineClass
from src.common.session import Session


class NoneBuffering(IBuffering, EngineClass):
    r"""
    no buffering, just trust the client vad audio data
    client(audio recoder)<--pipe-->serve(NLG)
    """

    TAG = "buffering_none"

    def __init__(self, **args) -> None:
        self.buffer = bytearray()

    def insert(self, audio_data):
        return self.buffer.extend(audio_data)

    def clear(self):
        self.buffer.clear()

    def is_voice_active(self, session) -> bool:
        if "vad_res" not in session.ctx.state or len(session.ctx.state["vad_res"]) == 0:
            return False
        return True

    def process_audio(self, session: Session):
        # Schedule the processing in a separate task on asyncio event loop
        asyncio.create_task(self.process_audio_async(session))

    async def process_audio_async(self, session: Session):
        start = time.time()
        if session.ctx.vad is not None:
            vad_res = await session.ctx.vad.detect(session)
            session.ctx.state["vad_res"] = vad_res

        if self.is_voice_active(session):
            session.ctx.asr_audio = self.buffer
            transcription = await session.ctx.asr.transcribe(session)
            if transcription["text"] != "":
                end = time.time()
                transcription["processing_time"] = end - start
            session.ctx.state["transcribe_res"] = transcription
