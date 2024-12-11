import logging
import random
from typing import AsyncGenerator
import uuid

from PIL import Image
from apipeline.frames.data_frames import Frame, TextFrame
from apipeline.pipeline.pipeline import FrameDirection

from src.processors.ai_processor import AIProcessor
from src.common.session import Session
from src.common.types import SessionCtx
from src.common.factory import EngineClass
from src.common.interface import IVisionDetector
from src.types.frames.data_frames import UserImageRawFrame


class DetectProcessor(AIProcessor):
    """
    input: image frame
    use detector like: YOLO, to process image frames, return text frame
    output: text frame
    """

    def __init__(
        self,
        detected_text: str | list = "welcome, my name is chat bot",
        out_detected_text: str | list = "goodbye, see you next time",
        detector: IVisionDetector | EngineClass | None = None,
        session: Session | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._detected_text = detected_text if isinstance(detected_text, list) else [detected_text]
        self._out_detected_text = (
            out_detected_text if isinstance(out_detected_text, list) else [out_detected_text]
        )
        self._detected = False
        self._detector = detector
        self._session = session
        if self._session is None:
            self._session = Session(**SessionCtx(uuid.uuid4()).__dict__)

    def set_detected_text(self, detected_text: str | list):
        self._detected_text = detected_text

    def set_out_detected_text(self, out_detected_text: str | list):
        self._out_detected_text = out_detected_text

    async def run_detect(self, frame: UserImageRawFrame) -> AsyncGenerator[Frame, None]:
        logging.debug(f"detect object from image: {frame}")
        if not frame.image:
            yield None
            return

        image = Image.frombytes(frame.mode, frame.size, frame.image)
        self._session.ctx.state["detect_img"] = image

        curr_detected = self._detector.detect(self._session)
        if curr_detected != self._detected:
            logging.info(f"current detected:{curr_detected}, dectected state:{self._detected}")
            self._detected = curr_detected
            if curr_detected and self._detected_text:
                yield TextFrame(text=random.choice(self._detected_text))
            if curr_detected is False and self._out_detected_text:
                yield TextFrame(text=random.choice(self._out_detected_text))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserImageRawFrame):
            await self.start_processing_metrics()
            await self.process_generator(self.run_detect(frame))
            await self.stop_processing_metrics()
        else:
            await self.push_frame(frame, direction)
