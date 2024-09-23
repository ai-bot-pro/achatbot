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


class VisionProcessor(VisionProcessorBase):
    """
    use vision lm to process image frames
    """

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

        logging.info(f"Analyzing image: {frame}")

        image = Image.frombytes(frame.mode, frame.size, frame.image)
        with BytesIO() as buffered:
            image.save(buffered, format=frame.format)
            img_base64_str = image_bytes_to_base64_data_uri(
                buffered.getvalue(), frame.format.lower())

        self._session.ctx.state["prompt"] = [
            {"type": "text", "text": frame.text},
        ]
        if "llm_transformers" in self._llm.SELECTED_TAG and \
                "vision" in self._llm.SELECTED_TAG:  # transformers vision
            self._session.ctx.state["prompt"].append(
                {"type": "image", "image": img_base64_str})
        else:  # llamacpp vision
            self._session.ctx.state["prompt"].append(
                {"type": "image_url", "image_url": {"url": img_base64_str}})

        iter = self._llm.chat_completion(self._session)
        for item in iter:
            yield TextFrame(text=item)


class MockVisionProcessor(VisionProcessorBase):
    """
    just mock
    """

    def __init__(
        self,
    ):
        super().__init__()

    async def run_vision(self, frame: VisionImageRawFrame) -> AsyncGenerator[Frame, None]:
        logging.info(f"Mock Analyzing image: {frame}")

        image = Image.frombytes(frame.mode, frame.size, frame.image)
        with BytesIO() as buffered:
            image.save(buffered, format=frame.format)
            img_base64_str = image_bytes_to_base64_data_uri(
                buffered.getvalue(), frame.format.lower())

        logging.info(f"Mock len(img_base64_str)+text: {len(img_base64_str)} {frame.text}")

        # await asyncio.sleep(1)

        yield TextFrame(text=f"你好！niubility, 你他娘的是个人才。请访问 github achatbot 进行把玩, 可以在colab中部署免费把玩。")
