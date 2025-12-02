import io
import os
import logging

import unittest
from PIL import Image

import aiohttp
from apipeline.frames.sys_frames import Frame
from apipeline.pipeline.pipeline import Pipeline, FrameProcessor
from apipeline.pipeline.task import PipelineTask, PipelineParams
from apipeline.pipeline.runner import PipelineRunner
from apipeline.frames.data_frames import TextFrame, ImageRawFrame
from apipeline.processors.frame_processor import FrameDirection

from src.processors.video import get_video_gen_processor
from src.common.logger import Logger
from apipeline.processors.logger import FrameLogger

from dotenv import load_dotenv

load_dotenv()


"""
mkdir -p images/videos

VIDEO_GEN_PROCESSOR=ComfyUIAPIVideoGenProcessor \
    API_URL=https://weedge--server-comfyui-api-dev.modal.run/video \
    MODEL=video_hunyuan_video_1_5_720p_t2v \
    python -m unittest test.integration.processors.test_video_gen_processor.TestProcessor
"""


class SaveVideoProcessor(FrameProcessor):
    def __init__(self, save_dir: str = "./images/videos/", **kwargs):
        super().__init__(**kwargs)
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, ImageRawFrame):
            print(f"save {str(frame)=} --> {self.save_dir}")
            from io import BytesIO

            img = Image.open(BytesIO(frame.image))
            img.save(self.save_dir + str(frame.id) + ".jpg", "JPEG")


class TestProcessor(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)
        cls.processor = os.getenv("VIDEO_GEN_PROCESSOR", "ComfyUIAPIVideoGenProcessor")
        cls.save_dir = os.getenv("SAVE_DIR_PATH", "./images/videos/")
        cls.device = os.getenv("DEVICE", "auto")
        cls.client_session = None

    @classmethod
    def tearDownClass(cls):
        pass

    def get_video_gen_processor_by_tag(self):
        kwargs = {}
        if self.processor == "ComfyUIAPIVideoGenProcessor":
            self.client_session = aiohttp.ClientSession()
            kwargs["aiohttp_session"] = self.client_session
            kwargs["width"] = 1280
            kwargs["height"] = 720
            kwargs["steps"] = 20
            kwargs["model"] = os.getenv("MODEL", "video_hunyuan_video_1_5_720p_t2v")

        return get_video_gen_processor(self.processor, **kwargs)

    async def asyncSetUp(self):
        pipeline = Pipeline(
            [
                self.get_video_gen_processor_by_tag(),
                FrameLogger(include_frame_types=[ImageRawFrame]),
                SaveVideoProcessor(save_dir=self.save_dir),
            ]
        )

        self.task = PipelineTask(pipeline, PipelineParams())

    async def asyncTearDown(self):
        self.client_session and await self.client_session.close()

    async def test_run(self):
        runner = PipelineRunner()
        await self.task.queue_frames(
            [
                TextFrame(
                    # text="A capybara holding a sign that reads Hello.",
                    text="一只熊猫在吃火锅，写实风格。",
                ),
            ]
        )
        await self.task.stop_when_done()
        await runner.run(self.task)
