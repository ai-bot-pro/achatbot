from abc import abstractmethod
from typing import AsyncGenerator

from apipeline.pipeline.pipeline import FrameDirection

from src.types.frames.data_frames import VisionImageRawFrame, Frame
from src.processors.ai_processor import AIProcessor


class VisionProcessorBase(AIProcessor):
    """VisionProcessorBase is a base class for vision processors."""

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
