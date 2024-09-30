import logging
import base64
from io import BytesIO
from typing import AsyncGenerator
import uuid

from PIL import Image
from apipeline.frames.data_frames import Frame, ImageRawFrame

from src.common.session import Session
from src.common.types import SessionCtx
from src.processors.vision.base import VisionProcessorBase
from src.common.factory import EngineClass
from src.common.interface import IVisionDetector
from src.types.frames.data_frames import VisionImageRawFrame


class AnnotateProcessor(VisionProcessorBase):
    """
    input: image frame
    use detector like: YOLO, to process image frames, annotate labels
    output: image frame
    """

    def __init__(
        self,
        detector: IVisionDetector | EngineClass | None = None,
        session: Session | None = None,
    ):
        super().__init__()
        self._detector = detector
        self._session = session
        if self._session is None:
            self._session = Session(**SessionCtx(uuid.uuid4()).__dict__)

    async def run_vision(self, frame: VisionImageRawFrame) -> AsyncGenerator[Frame, None]:
        logging.info(f"Annotating image: {frame}")
        if not frame.image:
            yield None
            return

        image = Image.frombytes(frame.mode, frame.size, frame.image)
        self._session.ctx.state["detect_img"] = image

        img_iter = self._detector.annotate(self._session)
        for img in img_iter:
            yield ImageRawFrame(
                image=img.tobytes(),
                mode=img.mode,
                size=img.size,
                format="JPEG",
            )
