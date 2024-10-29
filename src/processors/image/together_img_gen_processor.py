import base64
import io
import os
import logging
from typing import AsyncGenerator, Literal


try:
    from together import AsyncTogether
except ModuleNotFoundError as e:
    logging.error(
        "In order to use OpenAI, you need to `pip install together`. Also, set `TOGETHER_API_KEY` environment variable.")
    raise Exception(f"Missing module: {e}")
from PIL import Image
from apipeline.frames.data_frames import Frame, ImageRawFrame
from apipeline.frames.sys_frames import ErrorFrame

from src.processors.image.base import ImageGenProcessor


class TogetherImageGenProcessor(ImageGenProcessor):

    def __init__(
        self,
        *,
        width: int = 640,
        height: int = 480,
        model: str = "black-forest-labs/FLUX.1-schnell-Free",
        steps: int = 4,
    ):
        super().__init__()
        self._model = model
        self._width = width
        self._height = height
        self._steps = steps
        self._client = AsyncTogether()

    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        logging.debug(f"Generating image from prompt: {prompt}")

        image = await self._client.images.generate(
            prompt=prompt,
            model=self._model,
            height=self._height,
            width=self._width,
            steps=self._steps,
            n=1,
            response_format="b64_json",
        )

        b64_img = image.data[0].b64_json if len(image.data) > 0 else None

        if not b64_img:
            logging.error(f"{self} No image provided in response: {image}")
            yield ErrorFrame("Image generation failed")
            return

        image_bytes = base64.b64decode(b64_img)
        frame = ImageRawFrame(
            image=image_bytes,
            size=(self._width, self._height),
            format="",
            mode="",
        )
        yield frame
