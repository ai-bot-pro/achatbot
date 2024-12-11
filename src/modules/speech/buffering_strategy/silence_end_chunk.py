import os
import time
import asyncio
from common.interface import IBuffering
from common.session import Session
from common.types import SilenceAtEndOfChunkArgs


class SilenceAtEndOfChunk(IBuffering):
    def __init__(self, args: SilenceAtEndOfChunkArgs) -> None:
        self.chunk_length_seconds = os.environ.get("BUFFERING_CHUNK_LENGTH_SECONDS")
        if not self.chunk_length_seconds:
            self.chunk_length_seconds = args.chunk_length_seconds
        self.chunk_length_seconds = float(self.chunk_length_seconds)

        self.chunk_offset_seconds = os.environ.get("BUFFERING_CHUNK_OFFSET_SECONDS")
        if not self.chunk_offset_seconds:
            self.chunk_offset_seconds = args.chunk_offset_seconds
        self.chunk_offset_seconds = float(self.chunk_offset_seconds)

        self.processing_flag = False
        self.buffer = bytearray()
        self.scratch_buffer = bytearray()

    def insert(self, audio_data):
        self.buffer.extend(audio_data)

    def clear(self):
        self.buffer.clear()
        self.scratch_buffer.clear()

    def process_audio(self, session: Session):
        chunk_length_in_bytes = (
            self.chunk_length_seconds * session.ctx.sampling_rate * session.ctx.sample_width
        )
        if len(self.buffer) > chunk_length_in_bytes:
            if self.processing_flag:
                exit(
                    "Error in realtime processing: tried processing a new chunk while the previous one was still being processed"
                )

            self.scratch_buffer += session.buffer
            self.buffer.clear()
            self.processing_flag = True
            # Schedule the processing in a separate task on asyncio event loop
            asyncio.create_task(self.process_audio_async(session))

    def is_voice_active(self, session) -> bool:
        if "vad_res" not in session.ctx.state or len(session.ctx.state["vad_res"]) == 0:
            return False
        last_segment_should_end_before = (
            len(self.scratch_buffer) / (session.ctx.sampling_rate * session.ctx.sample_width)
        ) - self.chunk_offset_seconds
        return session.ctx.state["vad_res"][-1]["end"] < last_segment_should_end_before

    async def process_audio_async(self, session: Session):
        if session.ctx.vad is None or session.ctx.asr is None:
            self.processing_flag = False
            return

        start = time.time()
        vad_res = await session.ctx.vad.detect(session)
        session.ctx.state["vad_res"] = vad_res

        if self.is_voice_active(session):
            transcription = await session.ctx.asr.transcribe(session)
            if transcription["text"] != "":
                end = time.time()
                transcription["processing_time"] = end - start
            self.scratch_buffer.clear()

        self.processing_flag = False
