import io
import time
import wave
import logging
from abc import abstractmethod
from typing import AsyncGenerator

from apipeline.processors.frame_processor import FrameDirection, FrameProcessor
from apipeline.frames.sys_frames import CancelFrame, MetricsFrame, ErrorFrame
from apipeline.frames.control_frames import EndFrame
from apipeline.frames.data_frames import Frame, AudioRawFrame

from src.processors.speech.audio_volume_time_processor import AudioVolumeTimeProcessor
from src.processors.ai_processor import AsyncAIProcessor
from src.common.utils.helper import exp_smoothing, calculate_audio_volume
from src.types.frames.data_frames import DailyTransportMessageFrame, TranscriptionFrame
from src.types.speech.language import Language
from src.types.frames.control_frames import (
    ASRArgsUpdateFrame,
    ASRLanguageUpdateFrame,
    ASRModelUpdateFrame,
)


class ASRProcessorBase(AsyncAIProcessor):
    """ASRProcessorBase is a base class for speech-to-text processors."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    async def set_model(self, model: str):
        pass

    @abstractmethod
    async def set_language(self, language: Language):
        pass

    @abstractmethod
    async def set_asr_args(self, **args):
        pass

    @abstractmethod
    async def run_asr(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Returns transcript as a string"""
        pass

    async def process_audio_frame(self, frame: AudioRawFrame):
        await self.process_generator(self.run_asr(frame.audio))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Processes a frame of audio data, either buffering or transcribing it."""
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            # In this service we accumulate audio internally and at the end we
            # push a TextFrame. We don't really want to push audio frames down.
            await self.process_audio_frame(frame)
        elif isinstance(frame, ASRModelUpdateFrame):
            await self.set_model(frame.model)
        elif isinstance(frame, ASRLanguageUpdateFrame):
            await self.set_language(frame.language)
        elif isinstance(frame, ASRArgsUpdateFrame):
            await self.set_asr_args(frame.args)
        else:
            await self.push_frame(frame, direction)


class SegmentedASRProcessor(ASRProcessorBase):
    """SegmentedASRProcessor is a segement audio asr class for speech-to-text processors."""

    def __init__(
        self,
        *,
        min_volume: float = 0.6,
        max_silence_secs: float = 0.3,
        max_buffer_secs: float = 1.5,
        sample_rate: int = 16000,
        num_channels: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._min_volume = min_volume
        self._max_silence_secs = max_silence_secs
        self._max_buffer_secs = max_buffer_secs
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        (self._content, self._wave) = self._new_wave()
        self._silence_num_frames = 0
        # Volume exponential smoothing
        self._smoothing_factor = 0.2
        self._prev_volume = 0

    def _new_wave(self):
        content = io.BytesIO()
        ww = wave.open(content, "wb")
        ww.setsampwidth(2)
        ww.setnchannels(self._num_channels)
        ww.setframerate(self._sample_rate)
        return (content, ww)

    async def stop(self, frame: EndFrame):
        self._wave.close()

    async def cancel(self, frame: CancelFrame):
        self._wave.close()

    def _get_smoothed_volume(self, frame: AudioRawFrame) -> float:
        volume = calculate_audio_volume(frame.audio, frame.sample_rate)
        return exp_smoothing(volume, self._prev_volume, self._smoothing_factor)

    async def process_audio_frame(self, frame: AudioRawFrame):
        # Try to filter out empty background noise
        volume = self._get_smoothed_volume(frame)
        if volume >= self._min_volume and frame.audio and len(frame.audio) > 0:
            # If volume is high enough, write new data to wave file
            # Check if _wave is not None before writing
            if self._wave is not None:
                self._wave.writeframes(frame.audio)
            else:
                logging.error("Wave object is None, cannot write frames.")

            self._silence_num_frames = 0
        else:
            self._silence_num_frames += frame.num_frames
        self._prev_volume = volume

        # If buffer is not empty and we have enough data or there's been a long
        # silence, transcribe the audio gathered so far.
        silence_secs = self._silence_num_frames / self._sample_rate
        buffer_secs = self._wave.getnframes() / self._sample_rate
        if self._content.tell() > 0 and (
            buffer_secs > self._max_buffer_secs or silence_secs > self._max_silence_secs
        ):
            self._silence_num_frames = 0
            self._wave.close()
            self._content.seek(0)
            await self.start_processing_metrics()
            await self.process_generator(self.run_asr(self._content.read()))
            await self.stop_processing_metrics()
            (self._content, self._wave) = self._new_wave()


class TranscriptionTimingLogProcessor(FrameProcessor):
    """asr transcription timing log processor"""

    def __init__(self, avt: "AudioVolumeTimeProcessor"):
        super().__init__()
        self.name = "Transcription"
        self._avt = avt

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        try:
            await super().process_frame(frame, direction)
            if isinstance(frame, TranscriptionFrame):
                elapsed = time.time() - self._avt.last_transition_ts
                logging.info(f"Transcription TTF: {elapsed}")
                await self.push_frame(MetricsFrame(ttfb=[{self.name: elapsed}]))

            await self.push_frame(frame, direction)
        except Exception as e:
            logging.debug(f"Exception {e}")
