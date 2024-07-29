#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import io
import re
import wave
from abc import abstractmethod
from typing import AsyncGenerator

from apipeline.processors.async_frame_processor import AsyncFrameProcessor
from apipeline.processors.frame_processor import FrameDirection, FrameProcessor
from apipeline.frames.sys_frames import *
from apipeline.frames.control_frames import *
from apipeline.frames.data_frames import *

from src.common.utils.audio_utils import calculate_audio_volume, exp_smoothing
from src.types.frames.sys_frames import *
from src.types.frames.control_frames import *
from src.types.frames.data_frames import *


ENDOFSENTENCE_PATTERN_STR = r"""
    (?<![A-Z])       # Negative lookbehind: not preceded by an uppercase letter (e.g., "U.S.A.")
    (?<!\d)          # Negative lookbehind: not preceded by a digit (e.g., "1. Let's start")
    (?<!\d\s[ap])    # Negative lookbehind: not preceded by time (e.g., "3:00 a.m.")
    (?<!Mr|Ms|Dr)    # Negative lookbehind: not preceded by Mr, Ms, Dr (combined bc. length is the same)
    (?<!Mrs)         # Negative lookbehind: not preceded by "Mrs"
    (?<!Prof)        # Negative lookbehind: not preceded by "Prof"
    [\.\?\!:]        # Match a period, question mark, exclamation point, or colon
    $                # End of string
"""
ENDOFSENTENCE_PATTERN = re.compile(ENDOFSENTENCE_PATTERN_STR, re.VERBOSE)


def match_endofsentence(text: str) -> bool:
    return ENDOFSENTENCE_PATTERN.search(text.rstrip()) is not None


class AIProcessor(FrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def start(self, frame: StartFrame):
        pass

    async def stop(self, frame: EndFrame):
        pass

    async def cancel(self, frame: CancelFrame):
        pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.start(frame)
        elif isinstance(frame, CancelFrame):
            await self.cancel(frame)
        elif isinstance(frame, EndFrame):
            await self.stop(frame)

    async def process_generator(self, generator: AsyncGenerator[Frame, None]):
        async for f in generator:
            if isinstance(f, ErrorFrame):
                await self.push_error(f)
            else:
                await self.push_frame(f)


class AsyncAIProcessor(AsyncFrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def start(self, frame: StartFrame):
        pass

    async def stop(self, frame: EndFrame):
        pass

    async def cancel(self, frame: CancelFrame):
        pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.start(frame)
        elif isinstance(frame, CancelFrame):
            await self.cancel(frame)
        elif isinstance(frame, EndFrame):
            await self.stop(frame)


class LLMProcessor(AIProcessor):
    """This class is a no-op but serves as a base class for LLM processors."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._callbacks = {}
        self._start_callbacks = {}

    # TODO-CB: callback function type
    def register_function(self, function_name: str, callback, start_callback=None):
        self._callbacks[function_name] = callback
        if start_callback:
            self._start_callbacks[function_name] = start_callback

    def unregister_function(self, function_name: str):
        del self._callbacks[function_name]
        if self._start_callbacks[function_name]:
            del self._start_callbacks[function_name]

    def has_function(self, function_name: str):
        return function_name in self._callbacks.keys()

    async def call_function(self, function_name: str, args):
        if function_name in self._callbacks.keys():
            return await self._callbacks[function_name](self, args)
        return None

    async def call_start_function(self, function_name: str):
        if function_name in self._start_callbacks.keys():
            await self._start_callbacks[function_name](self)


class TTSProcessor(AIProcessor):
    def __init__(
            self,
            *,
            aggregate_sentences: bool = True,
            # if True, subclass is responsible for pushing TextFrames and LLMFullResponseEndFrames
            push_text_frames: bool = True,
            **kwargs):
        super().__init__(**kwargs)
        self._aggregate_sentences: bool = aggregate_sentences
        self._push_text_frames: bool = push_text_frames
        self._current_sentence: str = ""

    @abstractmethod
    async def set_voice(self, voice: str):
        pass

    # Converts the text to audio.
    @abstractmethod
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        pass

    async def say(self, text: str):
        await self.process_frame(TextFrame(text=text), FrameDirection.DOWNSTREAM)

    async def _handle_interruption(self, frame: StartInterruptionFrame, direction: FrameDirection):
        self._current_sentence = ""
        await self.push_frame(frame, direction)

    async def _process_text_frame(self, frame: TextFrame):
        text: str | None = None
        if not self._aggregate_sentences:
            text = frame.text
        else:
            self._current_sentence += frame.text
            if match_endofsentence(self._current_sentence):
                text = self._current_sentence
                self._current_sentence = ""

        if text:
            await self._push_tts_frames(text)

    async def _push_tts_frames(self, text: str, text_passthrough: bool = True):
        text = text.strip()
        if not text:
            return

        await self.push_frame(TTSStartedFrame())
        await self.start_processing_metrics()
        await self.process_generator(self.run_tts(text))
        await self.stop_processing_metrics()
        await self.push_frame(TTSStoppedFrame())
        if self._push_text_frames:
            # We send the original text after the audio. This way, if we are
            # interrupted, the text is not added to the assistant context.
            await self.push_frame(TextFrame(text))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            await self._process_text_frame(frame)
        elif isinstance(frame, StartInterruptionFrame):
            await self._handle_interruption(frame, direction)
        elif isinstance(frame, LLMFullResponseEndFrame) or isinstance(frame, EndFrame):
            sentence = self._current_sentence
            self._current_sentence = ""
            await self._push_tts_frames(sentence)
            if isinstance(frame, LLMFullResponseEndFrame):
                if self._push_text_frames:
                    await self.push_frame(frame, direction)
            else:
                await self.push_frame(frame, direction)
        elif isinstance(frame, TTSSpeakFrame):
            await self._push_tts_frames(frame.text, False)
        elif isinstance(frame, TTSVoiceUpdateFrame):
            await self.set_voice(frame.voice)
        else:
            await self.push_frame(frame, direction)


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


class ImageGenProcessor(AIProcessor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # Renders the image. Returns an Image object.
    @abstractmethod
    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            await self.push_frame(frame, direction)
            await self.start_processing_metrics()
            await self.process_generator(self.run_image_gen(frame.text))
            await self.stop_processing_metrics()
        else:
            await self.push_frame(frame, direction)


class VisionProcessor(AIProcessor):
    """VisionProcessor is a base class for vision processors."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._describe_text = None

    @abstractmethod
    async def run_vision(self, frame: VisionImageRawFrame) -> AsyncGenerator[Frame, None]:
        pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, VisionImageRawFrame):
            await self.start_processing_metrics()
            await self.process_generator(self.run_vision(frame))
            await self.stop_processing_metrics()
        else:
            await self.push_frame(frame, direction)
