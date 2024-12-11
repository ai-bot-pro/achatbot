import io
import os
import logging
from typing import AsyncGenerator, Literal


try:
    from openai import AsyncOpenAI
except ModuleNotFoundError as e:
    logging.error(
        "In order to use OpenAI, you need to `pip install openai`. Also, set `OPENAI_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")
import aiohttp
from PIL import Image
from apipeline.frames.data_frames import Frame
from apipeline.frames.sys_frames import ErrorFrame

from src.processors.image.base import ImageGenProcessor
from src.types.frames.data_frames import URLImageRawFrame


class OpenAIImageGenProcessor(ImageGenProcessor):
    def __init__(
        self,
        *,
        image_size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"],
        aiohttp_session: aiohttp.ClientSession,
        model: str = "dall-e-3",
    ):
        super().__init__()
        self._model = model
        self._image_size = image_size
        api_key = os.environ.get("OPENAI_API_KEY")
        self._client = AsyncOpenAI(api_key=api_key)
        self._aiohttp_session = aiohttp_session

    def set_size(self, width: int, height: int):
        self._image_size = f"{width}x{height}"

    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        logging.debug(f"Generating image from prompt: {prompt}")

        image = await self._client.images.generate(
            prompt=prompt, model=self._model, n=1, size=self._image_size
        )

        image_url = image.data[0].url if len(image.data) > 0 else None

        if not image_url:
            logging.error(f"{self} No image provided in response: {image}")
            yield ErrorFrame("Image generation failed")
            return

        # Load the image from the url
        async with self._aiohttp_session.get(image_url) as response:
            image_stream = io.BytesIO(await response.content.read())
            image = Image.open(image_stream).convert("RGB")
            frame = URLImageRawFrame(
                url=image_url,
                image=image.tobytes(),
                size=image.size,
                format=image.format if image.format else "JPEG",
                mode=image.mode,
            )
            yield frame
