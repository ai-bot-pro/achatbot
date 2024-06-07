import time
import asyncio

from common.interface import IBuffering
from common.session import Session


class NoneBuffering(IBuffering):

    def process_audio(self, session: Session):
        session.scratch_buffer += session.buffer
        session.buffer.clear()
        # Schedule the processing in a separate task on asyncio event loop
        asyncio.create_task(self.process_audio_async(session))

    async def process_audio(self, session: Session):
        if session.ctx.vad == None \
                or session.ctx.asr == None \
                or session.ctx.on_session_end == None:
            session.scratch_buffer.clear()
            session.buffer.clear()
            return

        start = time.time()
        vad_results = await session.ctx.vad.detect(session)
        if len(vad_results) == 0:
            session.scratch_buffer.clear()
            session.buffer.clear()
            return

        transcription = await session.ctx.asr.transcribe(session)
        if transcription['text'] != '':
            end = time.time()
            transcription['processing_time'] = end - start
            await session.ctx.on_session_end(transcription)
        session.scratch_buffer.clear()

        self.processing_flag = False
