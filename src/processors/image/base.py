from abc import abstractmethod
import logging
from typing import AsyncGenerator

from apipeline.pipeline.pipeline import FrameDirection

from src.processors.ai_processor import AIProcessor
from src.types.frames.data_frames import Frame, TextFrame


class ImageGenProcessor(AIProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._aiohttp_session = None
        self._width = 0
        self._height = 0
        self._gen_image_frame = TextFrame
        self._is_pass_frame = False

    def set_aiohttp_session(self, session):
        self._aiohttp_session = session

    def set_size(self, width: int, height: int):
        self._width = width
        self._height = height

    def set_gen_image_frame(self, frame: TextFrame):
        self._gen_image_frame = frame

    def set_is_pass_frame(self, is_pass: bool = False):
        self._is_pass_frame = is_pass

    @abstractmethod
    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, self._gen_image_frame):
            if self._is_pass_frame:
                await self.push_frame(frame, direction)
            await self.start_processing_metrics()
            await self.process_generator(self.run_image_gen(frame.text))
            await self.stop_processing_metrics()
        else:
            await self.push_frame(frame, direction)
