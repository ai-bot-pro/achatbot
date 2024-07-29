import io
import wave
from abc import abstractmethod
from typing import AsyncGenerator

from apipeline.processors.frame_processor import FrameDirection
from apipeline.frames.sys_frames import CancelFrame
from apipeline.frames.control_frames import EndFrame
from apipeline.frames.data_frames import Frame, AudioRawFrame

from src.processors.ai_processor import AIProcessor
from src.common.utils.audio_utils import calculate_audio_volume, exp_smoothing


class ASRProcessor(AIProcessor):
    """ASRProcessor is a base class for speech-to-text processors."""

    def __init__(self,
                 *,
                 min_volume: float = 0.6,
                 max_silence_secs: float = 0.3,
                 max_buffer_secs: float = 1.5,
                 sample_rate: int = 16000,
                 num_channels: int = 1,
                 **kwargs):
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

    @abstractmethod
    async def run_asr(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Returns transcript as a string"""
        pass

    def _new_wave(self):
        content = io.BytesIO()
        ww = wave.open(content, "wb")
        ww.setsampwidth(2)
        ww.setnchannels(self._num_channels)
        ww.setframerate(self._sample_rate)
        return (content, ww)

    def _get_smoothed_volume(self, frame: AudioRawFrame) -> float:
        volume = calculate_audio_volume(frame.audio, frame.sample_rate)
        return exp_smoothing(volume, self._prev_volume, self._smoothing_factor)

    async def _append_audio(self, frame: AudioRawFrame):
        # Try to filter out empty background noise
        volume = self._get_smoothed_volume(frame)
        if volume >= self._min_volume:
            # If volume is high enough, write new data to wave file
            self._wave.writeframes(frame.audio)
            self._silence_num_frames = 0
        else:
            self._silence_num_frames += frame.num_frames
        self._prev_volume = volume

        # If buffer is not empty and we have enough data or there's been a long
        # silence, transcribe the audio gathered so far.
        silence_secs = self._silence_num_frames / self._sample_rate
        buffer_secs = self._wave.getnframes() / self._sample_rate
        if self._content.tell() > 0 and (
                buffer_secs > self._max_buffer_secs or silence_secs > self._max_silence_secs):
            self._silence_num_frames = 0
            self._wave.close()
            self._content.seek(0)
            await self.start_processing_metrics()
            await self.process_generator(self.run_asr(self._content.read()))
            await self.stop_processing_metrics()
            (self._content, self._wave) = self._new_wave()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Processes a frame of audio data, either buffering or transcribing it."""
        await super().process_frame(frame, direction)

        if isinstance(frame, CancelFrame) or isinstance(frame, EndFrame):
            self._wave.close()
            await self.push_frame(frame, direction)
        elif isinstance(frame, AudioRawFrame):
            # In this processor we accumulate audio internally and at the end we
            # push a TextFrame. We don't really want to push audio frames down.
            await self._append_audio(frame)
        else:
            await self.push_frame(frame, direction)
