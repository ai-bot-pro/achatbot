import io
from abc import abstractmethod
from typing import AsyncGenerator, Optional

import wave
from apipeline.processors.frame_processor import FrameDirection
from apipeline.frames import Frame, AudioRawFrame, StartFrame

from src.processors.ai_processor import AsyncAIProcessor
from src.types.speech.language import Language
from src.types.frames.control_frames import (
    AvatarArgsUpdateFrame,
    AvatarLanguageUpdateFrame,
    AvatarModelUpdateFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)


class AvatarProcessorBase(AsyncAIProcessor):
    """AvatarProcessorBase is a base class for speech/text to motion processors."""

    def __init__(self, sample_rate=16000, **kwargs):
        super().__init__(**kwargs)
        self._sample_rate = sample_rate or 16000

    @abstractmethod
    async def set_model(self, model: str):
        pass

    @abstractmethod
    async def set_language(self, language: Language):
        pass

    @abstractmethod
    async def set_avatar_args(self, **args):
        pass

    @abstractmethod
    async def run_avatar(self, frame: AudioRawFrame) -> AsyncGenerator[Frame, None]:
        pass

    async def process_audio_frame(self, frame: AudioRawFrame):
        await self.process_generator(self.run_avatar(frame))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Processes a frame of audio data, either buffering or transcribing it."""
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            await self.process_audio_frame(frame)
        elif isinstance(frame, AvatarModelUpdateFrame):
            await self.set_model(frame.model)
        elif isinstance(frame, AvatarLanguageUpdateFrame):
            await self.set_language(frame.language)
        elif isinstance(frame, AvatarArgsUpdateFrame):
            await self.set_avatar_args(frame.args)
        else:
            await self.push_frame(frame, direction)


class SegmentedAvatarProcessor(AvatarProcessorBase):
    """SegmentedAvatarProcessor is an Avatar processor that uses VAD(user)/TTS(bot) events to detect
    speech and will run speech to motion(face,expressions,body) on speech segments only, instead of a
    continous stream. Since it uses VAD/TTS it means that VAD/TTS needs to be enabled in
    the pipeline.

    This processor always keeps a small audio buffer to take into account that VAD(user)/TTS(bot)
    events are delayed from when the user speech really starts.

    """

    def __init__(self, *, sample_rate: Optional[int] = None, **kwargs):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._content = None
        self._wave = None
        self._audio_buffer = bytearray()
        self._audio_buffer_size_1s = 0
        self._speaking = False

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._audio_buffer_size_1s = self._sample_rate * 2

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (UserStartedSpeakingFrame, TTSStartedFrame)):
            await self._handle_started_speaking(frame)
        elif isinstance(frame, (UserStoppedSpeakingFrame, TTSStoppedFrame)):
            await self._handle_stopped_speaking(frame)

    async def _handle_started_speaking(self, frame: UserStartedSpeakingFrame | TTSStartedFrame):
        self._speaking = True

    async def _handle_stopped_speaking(self, frame: UserStoppedSpeakingFrame | TTSStoppedFrame):
        self._speaking = False

        content = io.BytesIO()
        with wave.open(content, "wb") as wav:
            wav.setsampwidth(2)
            wav.setnchannels(1)
            wav.setframerate(self._sample_rate)
            wav.writeframes(self._audio_buffer)

        content.seek(0)
        await self.process_generator(
            self.run_avatar(
                AudioRawFrame(
                    audio=content.read(),
                    sample_rate=self._sample_rate,
                )
            )
        )

        self._audio_buffer.clear()

    async def process_audio_frame(self, frame: AudioRawFrame):
        # If the user is speaking the audio buffer will keep growing.
        self._audio_buffer += frame.audio

        # If the user is not speaking we keep just a little bit of audio.
        if not self._speaking and len(self._audio_buffer) > self._audio_buffer_size_1s:
            discarded = len(self._audio_buffer) - self._audio_buffer_size_1s
            self._audio_buffer = self._audio_buffer[discarded:]
