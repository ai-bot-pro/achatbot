import logging
import base64
from io import BytesIO
from typing import AsyncGenerator
import uuid

from PIL import Image
from apipeline.frames.sys_frames import ErrorFrame
from apipeline.frames.data_frames import Frame, TextFrame

from src.common.utils.img_utils import image_bytes_to_base64_data_uri
from src.common.session import Session
from src.common.types import SessionCtx
from src.processors.vision.base import VisionProcessorBase
from src.common.factory import EngineClass
from src.common.interface import ILlm
from src.types.frames.data_frames import VisionImageRawFrame


class LLamaCPPVisionProcessor(VisionProcessorBase):

    def __init__(
        self,
        llm: ILlm | EngineClass | None = None,
        session: Session | None = None,
    ):
        super().__init__()
        self._llm = llm
        self._session = session
        if self._session is None:
            self._session = Session(**SessionCtx(uuid.uuid4()).__dict__)

    def set_llm(self, llm: ILlm):
        self._llm = llm

    async def run_vision(self, frame: VisionImageRawFrame) -> AsyncGenerator[Frame, None]:
        """
        !TODO: image frame: PIL.Image, URL(str), base64 img(str) @weedge
        """
        if not self._llm:
            logging.error(f"{self} error: llm not available")
            yield ErrorFrame("llm not available")
            return

        logging.debug(f"Analyzing image: {frame}")

        image = Image.frombytes(frame.mode, frame.size, frame.image)
        with BytesIO() as buffered:
            image.save(buffered, format=frame.format)
            img_base64_str = image_bytes_to_base64_data_uri(
                buffered.getvalue(), frame.format.lower())

        self._session.ctx.state["prompt"] = [
            {"type": "text", "text": frame.text},
            {"type": "image_url", "image_url": {"url": img_base64_str}},
        ]

        iter = self._llm.chat_completion(self._session)
        for item in iter:
            yield TextFrame(text=item)
