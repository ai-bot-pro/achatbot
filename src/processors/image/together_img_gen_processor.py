import asyncio
import base64
import io
import os
import logging
import time
from typing import AsyncGenerator, Literal


try:
    from together import AsyncTogether
except ModuleNotFoundError as e:
    logging.error(
        "In order to use together ai, you need to `pip install achatbot[together_ai]`. Also, set `TOGETHER_API_KEY` environment variable."
    )
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
        gen_rate_s: int = 10,  # gen image per 10 s
    ):
        super().__init__()
        self._model = model
        self._width = width
        self._height = height
        self._steps = steps
        self._client = AsyncTogether()
        self._gen_rate_s = gen_rate_s
        self._pre_time_s = 0

    def set_aiohttp_session(self, session):
        pass

    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        logging.debug(f"Generating image from prompt: {prompt}")

        dur = int(time.time()) - self._pre_time_s
        if dur < self._gen_rate_s:
            await asyncio.sleep(self._gen_rate_s - dur)
        image = await self._client.images.generate(
            prompt=prompt,
            model=self._model,
            height=self._height,
            width=self._width,
            steps=self._steps,
            n=1,
            response_format="b64_json",
        )
        self._pre_time_s = int(time.time())

        b64_img = image.data[0].b64_json if len(image.data) > 0 else None

        if not b64_img:
            logging.error(f"{self} No image provided in response: {image}")
            yield ErrorFrame("Image generation failed")
            return

        image_bytes = base64.b64decode(b64_img)
        image_stream = io.BytesIO(image_bytes)
        image = Image.open(image_stream).convert("RGB")
        frame = ImageRawFrame(
            image=image.tobytes(),
            size=image.size,
            format=image.format if image.format else "JPEG",
            mode=image.mode,
        )
        yield frame
