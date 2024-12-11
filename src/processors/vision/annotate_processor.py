import logging
from typing import AsyncGenerator
import uuid

from PIL import Image
from apipeline.frames.data_frames import Frame
from apipeline.pipeline.pipeline import FrameDirection

from src.processors.ai_processor import AIProcessor
from src.common.session import Session
from src.common.types import SessionCtx
from src.common.factory import EngineClass
from src.common.interface import IVisionDetector
from src.types.frames.data_frames import UserImageRawFrame


class AnnotateProcessor(AIProcessor):
    """
    input: image frame
    use detector like: YOLO, to process image frames, annotate labels
    output: image frame
    """

    def __init__(
        self,
        detector: IVisionDetector | EngineClass | None = None,
        session: Session | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._detector = detector
        self._session = session
        if self._session is None:
            self._session = Session(**SessionCtx(uuid.uuid4()).__dict__)

    async def run_detect(self, frame: UserImageRawFrame) -> AsyncGenerator[Frame, None]:
        logging.debug(f"Annotating image: {frame}")
        if not frame.image:
            yield None
            return

        image = Image.frombytes(frame.mode, frame.size, frame.image)
        self._session.ctx.state["detect_img"] = image

        img_iter = self._detector.annotate(self._session)
        for img in img_iter:
            yield UserImageRawFrame(
                user_id=frame.user_id,
                image=img.tobytes(),
                mode=img.mode,
                size=img.size,
                format="JPEG",
            )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserImageRawFrame):
            await self.start_processing_metrics()
            await self.process_generator(self.run_detect(frame))
            await self.stop_processing_metrics()
        else:
            await self.push_frame(frame, direction)
