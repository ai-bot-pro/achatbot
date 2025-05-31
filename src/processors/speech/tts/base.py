import asyncio
import logging
import string
from abc import abstractmethod
from typing import AsyncGenerator

from apipeline.processors.frame_processor import FrameDirection, FrameProcessorMetrics
from apipeline.frames.control_frames import EndFrame
from apipeline.frames.sys_frames import Frame, MetricsFrame, StartInterruptionFrame
from apipeline.utils.string import match_endofsentence

from src.processors.ai_processor import AIProcessor
from src.types.frames.control_frames import (
    LLMFullResponseEndFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSVoiceUpdateFrame,
)
from src.types.frames.data_frames import TTSSpeakFrame, TextFrame


class TTSProcessorMetrics(FrameProcessorMetrics):
    async def start_tts_usage_metrics(self, text: str):
        characters = {
            "processor": self._name,
            "value": len(text),
        }
        logging.debug(f"{self._name} usage characters: {characters['value']}")
        return MetricsFrame(characters=[characters])


class TTSProcessorBase(AIProcessor):
    def __init__(
        self,
        *,
        aggregate_sentences: bool = True,
        # if True, subclass is responsible for pushing TextFrames and LLMFullResponseEndFrames
        push_text_frames: bool = True,
        sync_order_send: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._metrics = TTSProcessorMetrics(name=self.name)

        self._aggregate_sentences: bool = aggregate_sentences
        self._push_text_frames: bool = push_text_frames
        self._current_sentence: str = ""

        # sync event: tts done, step by step slow,
        # e.g. for test, push EndFrame in the end.
        self._tts_done_event = asyncio.Event()
        self._sync_order_send = sync_order_send

    def get_stream_info(self) -> dict:
        return {
            "sample_rate": 16000,
            "sample_width": 2,
            "channels": 1,
        }

    @abstractmethod
    async def set_voice(self, voice: str):
        pass

    @abstractmethod
    async def set_tts_args(self, **args):
        pass

    @abstractmethod
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        pass

    async def start_tts_usage_metrics(self, text: str):
        if self.can_generate_metrics() and self.usage_metrics_enabled:
            frame = await self._metrics.start_tts_usage_metrics(text)
            if frame:
                await self.push_frame(frame)

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
        # print(f"frame: {frame} text: {text}")
        if text:
            await self._push_tts_frames(text)

    async def _push_tts_frames(self, text: str, text_passthrough: bool = True):
        text = text.strip()
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        if not text:
            return

        # show text sentence before tts speak
        if self._push_text_frames:
            # We send the original text after the audio. This way, if we are
            # interrupted, the text is not added to the assistant context.
            await self.push_frame(TextFrame(text))

        await self.push_frame(TTSStartedFrame())
        await self.start_processing_metrics()

        # !NOTE: when open sync order send frame, u need set sync order
        # run_tts send audio frames over then send tts stopped frame
        # need subclass _tts_done_event to set;
        await self.process_generator(self.run_tts(text))
        if self._sync_order_send is True:
            await self._tts_done_event.wait()
            self._tts_done_event.clear()

        await self.stop_processing_metrics()
        await self.push_frame(TTSStoppedFrame())

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
