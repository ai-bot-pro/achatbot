import io
import os
import logging
from typing import AsyncGenerator

import aiohttp
from PIL import Image
from apipeline.frames.data_frames import Frame, ImageRawFrame
from apipeline.frames.sys_frames import ErrorFrame

from src.processors.image.base import ImageGenProcessor


class ComfyUIAPIImageGenProcessor(ImageGenProcessor):
    """
    ComfyUI API Video Gen Processor
    need cd deploy/modal to run src/comfyui/server.py
    ```
    MODEL_NAME=flux1_schnell_fp8 IMAGE_GPU=L40S modal serve src/comfyui/server.py
    ```
    """

    def __init__(
        self,
        *,
        aiohttp_session: aiohttp.ClientSession = None,
        api_url: str = "",
        model: str = "flux1_schnell_fp8",
        width: int = 1024,
        height: int = 1024,
        steps: int = 4,  # The number of steps used in the denoising process.
    ):
        super().__init__()
        self._model = model
        self._width = width
        self._height = height
        self._steps = steps
        self._aiohttp_session = aiohttp_session
        self._api_url = api_url or os.getenv("API_URL", "")

    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        logging.debug(f"Generating image from prompt: {prompt}")

        response = await self._aiohttp_session.post(
            self._api_url,
            json={
                "model": self._model,
                "prompt": prompt,
                "width": self._width,
                "height": self._height,
                "steps": self._steps,
            },
        )
        image_bytes = await response.content.read()

        if not image_bytes:
            yield ErrorFrame("Image generation failed")
            return

        image_stream = io.BytesIO(image_bytes)
        image = Image.open(image_stream).convert("RGB")
        # if image.size != (self._width, self._height):
        #    image = image.resize((self._width, self._height))
        frame = ImageRawFrame(
            image=image.tobytes(),
            size=image.size,
            format=image.format if image.format else "JPEG",
            mode=image.mode,
        )
        yield frame
