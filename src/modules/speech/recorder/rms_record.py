from typing import AsyncGenerator
import logging
import asyncio
import struct
import time


from src.common.interface import IRecorder
from src.common.audio_stream.helper import RingBuffer
from src.common.session import Session
from src.common.types import SILENCE_THRESHOLD, RATE
from .base import AudioRecorder


class RMSRecorder(AudioRecorder, IRecorder):
    TAG = "rms_recorder"

    def __init__(self, **args) -> None:
        super().__init__(**args)

    def compute_rms(self, data):
        # Assuming data is in 16-bit samples
        format = "<{}h".format(len(data) // 2)
        ints = struct.unpack(format, data)

        # Calculate RMS
        sum_squares = sum(i**2 for i in ints)
        rms = (sum_squares / len(ints)) ** 0.5
        return rms

    async def record_audio(self, session: Session) -> list[bytes | None]:
        frames = []
        async for item in self._record_audio_generator(session):
            item and frames.append(item)

        return frames

    async def record_audio_generator(self, session: Session) -> AsyncGenerator[bytes, None]:
        async for item in self._record_audio_generator(session):
            yield item

    async def _record_audio_generator(self, session: Session) -> AsyncGenerator[bytes, None]:
        if self.stream_info.in_sample_rate != RATE:
            raise Exception("rms recorder's sampling rate of the audio just support 16000Hz at now")
        silent_chunks = 0
        audio_started = False
        silence_timeout = 0
        if "silence_timeout_s" in session.ctx.state:
            logging.info(
                f"rms recording with silence_timeout {session.ctx.state['silence_timeout_s']} s"
            )
            silence_timeout = int(session.ctx.state["silence_timeout_s"])

        self.audio.start_stream()
        logging.debug(f"start {self.TAG} recording")
        start_time = time.time()
        if self.args.is_stream_callback is False:
            self.set_args(num_frames=self.stream_info.in_frames_per_buffer)
        while self.audio.is_stream_active():
            data = self.get_record_buf()
            if len(data) == 0:
                await asyncio.sleep(self.args.no_stream_sleep_time_s)
                continue
            rms = self.compute_rms(data)
            if audio_started:
                yield data
                if rms < SILENCE_THRESHOLD:
                    silent_chunks += 1
                    if silent_chunks > self.args.silent_chunks:
                        yield None
                        break
                else:
                    silent_chunks = 0
            elif rms >= SILENCE_THRESHOLD:
                yield data
                audio_started = True
            else:
                if silence_timeout > 0 and time.time() - start_time > silence_timeout:
                    logging.warning("rms recording silence timeout")
                    break

        self.audio.stop_stream()
        logging.debug(f"end {self.TAG} recording")


class WakeWordsRMSRecorder(RMSRecorder):
    TAG = "wakeword_rms_recorder"

    def __init__(self, **args) -> None:
        super().__init__(**args)

    async def _record_audio(self, session: Session) -> list[bytes]:
        sample_rate, frame_length = session.ctx.waker.get_sample_info()
        self.sample_rate, self.frame_length = sample_rate, frame_length
        self.set_args(frames_per_buffer=self.frame_length)

        # ring buffer
        pre_recording_buffer_duration = 3.0
        maxlen = int((sample_rate // frame_length) * pre_recording_buffer_duration)
        self.audio_buffer = RingBuffer(maxlen)

        self.audio.start_stream()
        logging.info(
            f"start {self.TAG} recording; audio sample_rate: {self.sample_rate},frame_length:{self.frame_length}, audio buffer maxlen: {maxlen}"
        )

        if self.args.is_stream_callback is False:
            self.set_args(num_frames=self.frame_length)

        while True:
            data = self.get_record_buf()
            if len(data) == 0:
                await asyncio.sleep(self.args.no_stream_sleep_time_s)
                continue
            session.ctx.read_audio_frames = data
            session.ctx.waker.set_audio_data(self.audio_buffer.get_buf())
            res = await session.ctx.waker.detect(session)
            if res is True:
                break
            self.audio_buffer.extend(data)

        self.audio.stop_stream()
        logging.debug("end wake words detector rms recording")

        if self.args.silence_timeout_s > 0:
            session.ctx.state["silence_timeout_s"] = self.args.silence_timeout_s

    async def record_audio(self, session: Session) -> list[bytes | None]:
        if session.ctx.waker is None:
            logging.warning(f"{self.TAG} no waker instance in session ctx, use {super().TAG}")
            return await super().record_audio(session)

        await self._record_audio(session)
        return await super().record_audio(session)

    async def record_audio_generator(self, session: Session) -> AsyncGenerator[bytes, None]:
        if session.ctx.waker is None:
            logging.warning(f"{self.TAG} no waker instance in session ctx, use {super().TAG}")
            async for item in super().record_audio_generator(session):
                yield item

        await self._record_audio(session)
        async for item in super().record_audio_generator(session):
            yield item
