from dataclasses import dataclass
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
        aiohttp_session: aiohttp.ClientSession = None,
        model: str = "stabilityai/stable-diffusion-3.5-large",
        width: int = 1024,
        height: int = 1024,
        steps: int = 28,
    ):
        super().__init__()
        self._model = model
        self._width = width
        self._height = height
        self._steps = steps
        self._api_key = os.environ.get("HF_API_KEY")
        self._aiohttp_session = aiohttp_session

    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        logging.debug(f"Generating image from prompt: {prompt}")

        API_URL = f"https://api-inference.huggingface.co/models/{self._model}"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        response = await self._aiohttp_session.post(
            API_URL,
            headers=headers,
            json={
                "inputs": prompt,
                # target_size is not work, use default params
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
