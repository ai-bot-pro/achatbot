import os
import time
import asyncio
from common.interface import IBuffering
from common.session import Session
from common.types import SilenceAtEndOfChunkArgs


class SilenceAtEndOfChunk(IBuffering):
    def __init__(self, args: SilenceAtEndOfChunkArgs) -> None:
        self.chunk_length_seconds = os.environ.get(
            'BUFFERING_CHUNK_LENGTH_SECONDS')
        if not self.chunk_length_seconds:
            self.chunk_length_seconds = args.chunk_length_seconds
        self.chunk_length_seconds = float(self.chunk_length_seconds)

        self.chunk_offset_seconds = os.environ.get(
            'BUFFERING_CHUNK_OFFSET_SECONDS')
        if not self.chunk_offset_seconds:
            self.chunk_offset_seconds = args.chunk_offset_seconds
        self.chunk_offset_seconds = float(self.chunk_offset_seconds)

        self.processing_flag = False

    def process_audio(self, session: Session):
        chunk_length_in_bytes = self.chunk_length_seconds * \
            session.args.sampling_rate * session.args.samples_width
        if len(session.buffer) > chunk_length_in_bytes:
            if self.processing_flag:
                exit("Error in realtime processing: tried processing a new chunk while the previous one was still being processed")

            session.scratch_buffer += session.buffer
            session.buffer.clear()
            self.processing_flag = True
            # Schedule the processing in a separate task on asyncio event loop
            asyncio.create_task(self.process_audio_async(session))

    async def process_audio_async(self, session: Session):
        if session.args.vad == None \
                or session.args.asr == None \
                or session.args.on_session_end == None:
            session.scratch_buffer.clear()
            session.buffer.clear()
            self.processing_flag = False
            return

        start = time.time()
        vad_results = await session.args.vad.detect(session)
        if len(vad_results) == 0:
            session.scratch_buffer.clear()
            session.buffer.clear()
            return

        last_segment_should_end_before = ((len(session.scratch_buffer) / (
            session.args.sampling_rate * session.args.samples_width)) - self.chunk_offset_seconds)
        if vad_results[-1]['end'] < last_segment_should_end_before:
            transcription = await session.args.asr.transcribe(session)
            if transcription['text'] != '':
                end = time.time()
                transcription['processing_time'] = end - start
                await session.args.on_session_end(transcription)
            session.scratch_buffer.clear()

        self.processing_flag = False
