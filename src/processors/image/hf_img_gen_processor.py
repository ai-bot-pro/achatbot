import io
import os
import logging
from typing import AsyncGenerator

import aiohttp
from PIL import Image
from apipeline.frames.data_frames import Frame, ImageRawFrame
from apipeline.frames.sys_frames import ErrorFrame

from src.processors.image.base import ImageGenProcessor


# https://huggingface.co/docs/api-inference/tasks/text-to-image

class HFApiInferenceImageGenProcessor(ImageGenProcessor):

    def __init__(
        self,
        *,
        aiohttp_session: aiohttp.ClientSession,
        api_key: str,
        model: str = "stabilityai/stable-diffusion-3.5-large",
        width: int = 640,
        height: int = 480,
        steps: int = 28,
    ):
        super().__init__()
        self._model = model
        self._width = width
        self._height = height
        self._steps = steps
        self._api_key = os.environ.get("HF_API_KEY", api_key)
        self._aiohttp_session = aiohttp_session

    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        logging.debug(f"Generating image from prompt: {prompt}")

        API_URL = f"https://api-inference.huggingface.co/models/{self._model}"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        response = await self._aiohttp_session.post(
            API_URL, headers=headers,
            json={
                "inputs": prompt,
                # "target_size": {
                #    "width": self._width,
                #    "height": self._height,
                # },
                # "num_inference_steps": self._steps,
            },
        )
        image_bytes = await response.content.read()

        if not image_bytes:
            yield ErrorFrame("Image generation failed")
            return

        frame = ImageRawFrame(
            image=image_bytes,
            size=(1024, 1024),
            format="",
            mode="",
        )
        yield frame
