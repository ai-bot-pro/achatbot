import logging
import base64
from io import BytesIO
from typing import AsyncGenerator
import uuid
import asyncio

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
    input: image frame
    use vision lm to process image frames
    output: text frame
    """

    def __init__(
        self,
        llm: ILlm | EngineClass | None = None,
        session: Session | None = None,
        sleep_time_s: float = 0.15,
    ):
        super().__init__()
        self._llm = llm
        self._session = session
        if self._session is None:
            self._session = Session(**SessionCtx(uuid.uuid4()).__dict__)
        self.sleep_time_s = sleep_time_s

    def set_llm(self, llm: ILlm):
        self._llm = llm

    async def run_vision(self, frame: VisionImageRawFrame) -> AsyncGenerator[Frame, None]:
        if not self._llm:
            logging.error(f"{self} error: llm not available")
            yield ErrorFrame("llm not available")
            return

        logging.debug(f"Analyzing image: {frame}")

        if (
            "llm_transformers" in self._llm.SELECTED_TAG
            and "vision_janus" in self._llm.SELECTED_TAG
        ):  # transformers vision janus pro
            async for item in self._run_imgs_text_vision(frame):
                yield item
        elif (
            "llm_transformers" in self._llm.SELECTED_TAG
            and "vision_deepseek" in self._llm.SELECTED_TAG
        ):  # transformers vision deepseekvl2
            async for item in self._run_imgs_text_vision(frame):
                yield item
        elif (
            "llm_transformers" in self._llm.SELECTED_TAG
            and "vision_minicpmo" in self._llm.SELECTED_TAG
        ):  # transformers vision MiniCPM-o
            async for item in self._run_imgs_text_vision(frame):
                yield item
        elif (
            "llm_transformers" in self._llm.SELECTED_TAG
            and "vision_fastvlm" in self._llm.SELECTED_TAG
        ):  # transformers vision FastVLM
            async for item in self._run_imgs_text_vision(frame):
                yield item
        else:  # gemma3, smolvlm, qwen vision (kimi vision) is default, nice vision prompt
            async for item in self._run_vision(frame):
                yield item

    async def _run_imgs_text_vision(
        self, frame: VisionImageRawFrame
    ) -> AsyncGenerator[Frame, None]:
        self._session.ctx.state["prompt"] = []
        if frame.image:
            image = Image.frombytes(frame.mode, frame.size, frame.image)
            self._session.ctx.state["prompt"].append(image)
        self._session.ctx.state["prompt"].append(frame.text)

        iter = self._llm.chat_completion(self._session)
        for item in iter:
            if item is None:
                await asyncio.sleep(self.sleep_time_s)
                continue
            yield TextFrame(text=item)

    async def _run_vision(self, frame: VisionImageRawFrame) -> AsyncGenerator[Frame, None]:
        self._session.ctx.state["prompt"] = [
            {"type": "text", "text": frame.text},
        ]

        if frame.image:
            image = Image.frombytes(frame.mode, frame.size, frame.image)
            with BytesIO() as buffered:
                image.save(buffered, format=frame.format)
                img_base64_str = image_bytes_to_base64_data_uri(
                    buffered.getvalue(), frame.format.lower()
                )

            if (
                "llm_transformers" in self._llm.SELECTED_TAG and "vision" in self._llm.SELECTED_TAG
            ):  # transformers vision
                self._session.ctx.state["prompt"].append({"type": "image", "image": img_base64_str})
            elif (
                "llm_fastdeploy" in self._llm.SELECTED_TAG and "vision" in self._llm.SELECTED_TAG
            ):  # fastdeploy vision
                self._session.ctx.state["prompt"].append({"type": "image_url", "image_url": image})
            else:  # llamacpp vision
                self._session.ctx.state["prompt"].append(
                    {"type": "image_url", "image_url": {"url": img_base64_str}}
                )

        iter = self._llm.chat_completion(self._session)
        for item in iter:
            if item is None:  # yield coroutine to speak
                await asyncio.sleep(self.sleep_time_s)
                continue
            yield TextFrame(text=item)


class MockVisionProcessor(VisionProcessorBase):
    """
    just mock
    """

    def __init__(
        self,
        mock_text: str = "你好！niubility。"
        "你他娘的是个人才。"
        "请访问 github achatbot 进行把玩, 可以在colab中部署免费把玩。",
    ):
        super().__init__()
        self._mock_text = mock_text

    async def run_vision(self, frame: VisionImageRawFrame) -> AsyncGenerator[Frame, None]:
        logging.debug(f"Mock Analyzing image: {frame}")

        if frame.image:
            image = Image.frombytes(frame.mode, frame.size, frame.image)
            with BytesIO() as buffered:
                image.save(buffered, format=frame.format)
                img_base64_str = image_bytes_to_base64_data_uri(
                    buffered.getvalue(), frame.format.lower()
                )

            logging.info(f"Mock len(img_base64_str)+text: {len(img_base64_str)} {frame.text}")

        # await asyncio.sleep(1)

        yield TextFrame(text=self._mock_text)
