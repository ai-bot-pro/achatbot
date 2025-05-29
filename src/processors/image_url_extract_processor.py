import asyncio
import logging
import io
import re

from PIL import Image
import aiohttp
from apipeline.processors.frame_processor import FrameDirection, FrameProcessor

from src.types.frames.data_frames import (
    Frame,
    FunctionCallResultFrame,
    URLImageRawFrame,
)


class UrlToImageProcessor(FrameProcessor):
    def __init__(self, aiohttp_session: aiohttp.ClientSession, **kwargs):
        super().__init__(**kwargs)
        self._aiohttp_session = aiohttp_session

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, FunctionCallResultFrame):
            await self.push_frame(frame, direction)
            image_url = self.extract_url(frame.result)
            if image_url:
                await self.run_image_process(image_url)
                # sometimes we get multiple image urls- process 1 at a time
                await asyncio.sleep(1)
        else:
            await self.push_frame(frame, direction)

    def extract_url(self, text: str):
        pattern = r"!\[[^\]]*\]\((https?://[^)]+\.(png|jpg|jpeg|PNG|JPG|JPEG))\)"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        return None

    async def run_image_process(self, image_url: str):
        try:
            async with self._aiohttp_session.get(image_url) as response:
                image_stream = io.BytesIO(await response.content.read())
                image = Image.open(image_stream)
                image = image.convert("RGB")
                logging.info(f"handling image from url: '{image_url}' | image size: {image.size}")
                frame = URLImageRawFrame(
                    url=image_url,
                    image=image.tobytes(),
                    size=image.size,
                    format=image.format,
                    mode=image.mode,
                )
                await self.push_frame(frame)
        except Exception as e:
            error_msg = f"Error handling image url {image_url}: {str(e)}"
            logging.error(error_msg)
