import logging
from typing import AsyncGenerator
import uuid

from PIL import Image
from apipeline.frames.data_frames import Frame, TextFrame
from apipeline.pipeline.pipeline import FrameDirection

from src.processors.ai_processor import AIProcessor
from src.common.session import Session
from src.common.types import SessionCtx
from src.common.factory import EngineClass
from src.common.interface import IVisionOCR
from src.types.frames.data_frames import UserImageRawFrame


class OCRProcessor(AIProcessor):
    """
    input: image frame
    use OCR, to process image frames, return text frame
    output: text frame
    """

    def __init__(
        self, ocr: IVisionOCR | EngineClass | None = None, session: Session | None = None, **kwargs
    ):
        super().__init__(**kwargs)
        self._detected = False
        self._ocr = ocr
        self._session = session
        if self._session is None:
            self._session = Session(**SessionCtx(uuid.uuid4()).__dict__)

    async def run_detect(self, frame: UserImageRawFrame) -> AsyncGenerator[Frame, None]:
        logging.debug(f"OCR from image: {frame}")
        if not frame.image:
            yield None
            return

        image = Image.frombytes(frame.mode, frame.size, frame.image)
        self._session.ctx.state["ocr_img"] = image

        detect_iter = self._ocr.generate(self._session)
        for item in detect_iter:
            yield TextFrame(text=item)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserImageRawFrame):
            await self.start_processing_metrics()
            await self.process_generator(self.run_detect(frame))
            await self.stop_processing_metrics()
        else:
            await self.push_frame(frame, direction)
