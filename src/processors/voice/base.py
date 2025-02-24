from abc import abstractmethod
import io
import logging
from typing import AsyncGenerator
import wave

from apipeline.frames import Frame, AudioRawFrame, EndFrame, CancelFrame, TextFrame
from apipeline.processors.frame_processor import FrameDirection, FrameProcessor

from src.common.types import CHANNELS, RATE
from src.common.utils.helper import calculate_audio_volume, exp_smoothing
from src.processors.ai_processor import AsyncAIProcessor


class VoiceProcessorBase(AsyncAIProcessor):
    """
    VoiceProcessorBase is a base class for voice processors
    (just become a Monolith model with adpter and prompt to gen speech and text). e.g.:
    - A1-T1: (speech)-to-(text) (asr)
    - A1-T2: (speech)-to-(text) (audio gen/chat to text)
    - T1-T2A2: (text)-to-(speech and text) ((text-llm)+tts) (text gen/chat to text/audio)
    - A1-T2A2: (speech)-to-(speech and text) (asr+(text-llm)+tts) (audio gen/chat to text/audio)
    - T1-A1: (text)-to-(speech) (tts)
    - T1-A2: (text)-to-(speech) (text gen/chat to audio)

    support streaming mode
    # TODO: function call
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def stream_info(self) -> dict:
        """Return dict out stream info"""
        return {"sample_rate": RATE, "channels": CHANNELS}

    async def run_voice(self, frame: AudioRawFrame) -> AsyncGenerator[Frame, None]:
        """
        yield AudioRawFrame | AudioRawFrame + TextFrame async generator or None (internal push/queue frame)
        """
        yield frame

    async def run_text(self, frame: TextFrame) -> AsyncGenerator[Frame, None]:
        """
        yield AudioRawFrame | AudioRawFrame + TextFrame async generator or None (internal push/queue frame)
        """
        yield frame

    async def process_text_frame(self, frame: TextFrame):
        await self.process_generator(self.run_text(frame))

    async def process_audio_frame(self, frame: AudioRawFrame):
        await self.process_generator(self.run_voice(frame))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Processes a frame of audio data, either buffering or transcribing it."""
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            await self.process_text_frame(frame)
        elif isinstance(frame, AudioRawFrame):
            await self.process_audio_frame(frame)
        else:
            await self.push_frame(frame, direction)


class SegmentedVoiceProcessor(VoiceProcessorBase):
    """SegmentedVoiceProcessor is a segement audio voice class for speech-to-speech processors."""

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
            await self.process_generator(
                self.run_voice(
                    AudioRawFrame(
                        audio=self._content.read(),
                        sample_rate=self._sample_rate,
                        num_channels=self._num_channels,
                        sample_width=2,
                    )
                )
            )
            await self.stop_processing_metrics()
            (self._content, self._wave) = self._new_wave()
