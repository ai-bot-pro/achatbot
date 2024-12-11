from typing import AsyncGenerator
import collections
import logging
import time
import asyncio

from src.common.interface import IRecorder
from src.common.audio_stream.helper import RingBuffer
from src.common.session import Session
from src.common.types import VADRecoderArgs, AudioRecoderArgs
from .base import AudioRecorder


class VADRecorder(AudioRecorder, IRecorder):
    TAG = "vad_recorder"

    def __init__(self, **args) -> None:
        vad_args = VADRecoderArgs(**args)
        super().__init__(
            **AudioRecoderArgs(
                silent_chunks=vad_args.silent_chunks,
                silence_timeout_s=vad_args.silence_timeout_s,
                num_frames=vad_args.num_frames,
                is_stream_callback=vad_args.is_stream_callback,
                no_stream_sleep_time_s=vad_args.no_stream_sleep_time_s,
            ).__dict__
        )
        self.args = vad_args

    async def record_audio(self, session: Session) -> list[bytes | None]:
        frames = []
        async for item in self._record_audio_generator(session):
            item and frames.append(item)

        return frames

    async def record_audio_generator(self, session: Session) -> AsyncGenerator[bytes, None]:
        async for item in self._record_audio_generator(session):
            yield item

    async def _record_audio_generator(self, session: Session) -> AsyncGenerator[bytes, None]:
        if session.ctx.vad is None:
            raise Exception("VAD is not set")

        silence_timeout = 0
        if "silence_timeout_s" in session.ctx.state:
            logging.info(
                f"{self.TAG} recording with silence_timeout {session.ctx.state['silence_timeout_s']} s"
            )
            silence_timeout = int(session.ctx.state["silence_timeout_s"])

        frames_duration_ms = (
            1000 * self.stream_info.in_frames_per_buffer // self.stream_info.in_sample_rate
        )
        num_padding_frames = self.args.padding_ms // frames_duration_ms
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False

        self.audio.start_stream()
        logging.debug(f"start {self.TAG} recording")
        start_time = time.time()
        if self.args.is_stream_callback is False:
            self.set_args(num_frames=self.stream_info.in_frames_per_buffer)

        for frame in self.frame_genrator():
            if frame is None:
                yield None
                continue
            if len(frame) == 0:
                await asyncio.sleep(self.args.no_stream_sleep_time_s)
                continue
            session.ctx.vad.set_audio_data(frame)
            is_active_speech = await session.ctx.vad.detect(session)
            if not triggered:
                ring_buffer.append((frame, is_active_speech))
                num_voiced = len([f for f, is_active_speech in ring_buffer if is_active_speech])
                # logging.debug( f"num_voiced:{num_voiced} threshold: {self.args.active_ratio} * {ring_buffer.maxlen}")
                if num_voiced > self.args.active_ratio * ring_buffer.maxlen:
                    triggered = True
                    for f, is_active_speech in ring_buffer:
                        if is_active_speech:
                            yield f
                    ring_buffer.clear()
                else:
                    if silence_timeout > 0 and time.time() - start_time > silence_timeout:
                        logging.warning(f"{self.TAG} recording silence timeout")
                        break
            else:
                yield frame
                ring_buffer.append((frame, is_active_speech))
                num_unvoiced = len(
                    [f for f, is_active_speech in ring_buffer if not is_active_speech]
                )
                logging.debug(
                    f"num_unvoiced: {num_unvoiced} threshold: {self.args.silent_ratio} * {ring_buffer.maxlen}"
                )
                if num_unvoiced > self.args.silent_ratio * ring_buffer.maxlen:
                    triggered = False
                    yield None
                    ring_buffer.clear()
                    break
            # end if
        # end for

        self.audio.stop_stream()
        logging.debug(f"end {self.TAG} recording")


class WakeWordsVADRecorder(VADRecorder):
    TAG = "wakeword_vad_recorder"

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

    async def record_audio(self, session: Session) -> list[bytes]:
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
