import cv2
import os
import datetime
import time
import io
import logging
import asyncio
import tempfile
from typing import AsyncGenerator

import aiohttp
from PIL import Image
from apipeline.frames.data_frames import Frame, ImageRawFrame
from apipeline.frames.sys_frames import ErrorFrame

from src.processors.video.base import VideoGenProcessor


class ComfyUIAPIVideoGenProcessor(VideoGenProcessor):
    """
    ComfyUI API Video Gen Processor
    need cd deploy/modal to run src/comfyui/server.py
    ```
    MODEL_NAME=video_hunyuan_video_1_5_720p_t2v IMAGE_GPU=L40S modal serve src/comfyui/server.py
    ```
    """

    def __init__(
        self,
        *,
        aiohttp_session: aiohttp.ClientSession = None,
        api_url: str = "",
        model: str = "video_hunyuan_video_1.5_720p_t2v",
        width: int = 1024,
        height: int = 1024,
        length: int = 121,
        codec: str = "h264",
        steps: int = 20,  # The number of steps used in the denoising process.
    ):
        super().__init__()
        self._model = model
        self._width = width
        self._height = height
        self._length = length
        self._codec = codec
        self._steps = steps
        self._aiohttp_session = aiohttp_session
        self._api_url = api_url or os.getenv("API_URL", "")

    async def run_video_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        logging.debug(f"Generating image from prompt: {prompt}")

        async with self._aiohttp_session.post(
            self._api_url,
            json={
                "model": self._model,
                "prompt": prompt,
                "width": self._width,
                "height": self._height,
                "length": self._length,
                "codec": self._codec,
                "steps": self._steps,
            },
        ) as response:
            if response.status != 200:
                error_msg = await response.text()
                logging.error(f"Video generation failed: {error_msg}")
                yield ErrorFrame(f"Video generation failed: {error_msg}")
                return

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                temp_file_path = temp_file.name
                async for chunk in response.content.iter_chunked(1024 * 1024):
                    temp_file.write(chunk)

        try:
            async for frame in self.generate_frames_from_video(temp_file_path):
                yield frame
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    async def generate_frames_from_video(self, video_source) -> AsyncGenerator[Frame, None]:
        """
        Generator function to read frames from a video source (path or URL) and yield them as JPEG bytes.
        The frames are yielded at a rate consistent with the original video's FPS.
        """
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            logging.error(f"Error: Could not open video source {video_source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logging.debug(f"{fps=},{total_frames=}")

        # Calculate the delay needed between frames to match the original video's FPS
        # If fps is 0 (e.g., for some image streams), avoid division by zero
        delay_s = 1.0 / fps if fps > 0 else 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if len(frame.shape) != 3:
                logging.error("Frame is not a 3-channel image")
                break
            height, width, channel = frame.shape
            if channel != 3:
                logging.error("Frame is not a 3-channel image")
                break

            # OpenCV captures frames in BGR format by default.
            # If the intention is to convert to RGB before encoding, perform the conversion.
            # This check assumes 3-channel images from video capture are BGR and need conversion to RGB.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Encode the frame as JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]  # JPEG quality 90
            _, buffer = cv2.imencode(".jpg", frame, encode_param)
            frame_bytes = buffer.tobytes()

            image_frame = ImageRawFrame(
                image=frame_bytes,
                size=(width, height),
                format="JPEG",
                mode="RGB",
            )
            yield image_frame

            if delay_s > 0:
                await asyncio.sleep(delay_s)

        cap.release()
        logging.info(f"Finished streaming frames from {video_source}")
