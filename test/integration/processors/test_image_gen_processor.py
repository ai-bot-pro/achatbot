import io
import os
import logging
import uuid

import unittest
from PIL import Image

import aiohttp
from apipeline.frames.sys_frames import Frame
from apipeline.pipeline.pipeline import Pipeline, FrameProcessor
from apipeline.pipeline.task import PipelineTask, PipelineParams
from apipeline.pipeline.runner import PipelineRunner
from apipeline.frames.data_frames import TextFrame, ImageRawFrame
from apipeline.processors.frame_processor import FrameDirection

from src.common.types import SRC_PATH
from src.common.logger import Logger
from apipeline.processors.logger import FrameLogger

from dotenv import load_dotenv
load_dotenv()


"""
IMAGE_GEN_PROCESSOR=HFApiInferenceImageGenProcessor \
    python -m unittest test.integration.processors.test_image_gen_processor.TestProcessor

IMAGE_GEN_PROCESSOR=OpenAIImageGenProcessor \
    python -m unittest test.integration.processors.test_image_gen_processor.TestProcessor

IMAGE_GEN_PROCESSOR=TogetherImageGenProcessor \
    python -m unittest test.integration.processors.test_image_gen_processor.TestProcessor
"""


class SaveImageProcessor(FrameProcessor):
    def __init__(self, save_file: str = "./images/test_gen.jpeg", **kwargs):
        super().__init__(**kwargs)
        self.save_file = save_file

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, ImageRawFrame):
            print("save-->", frame, self.save_file)
            img = Image.open(io.BytesIO(frame.image))
            img.save(self.save_file, frame.format)


class TestProcessor(unittest.IsolatedAsyncioTestCase):

    @classmethod
    def setUpClass(cls):
        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)
        cls.processor = os.getenv("IMAGE_GEN_PROCESSOR", "HFApiInferenceImageGenProcessor")
        cls.save_img_file = os.getenv("SAVE_FILE_PATH", "./images/test_gen.jpeg")
        cls.client_session = None

    @classmethod
    def tearDownClass(cls):
        pass

    def get_image_gen_processor(self):
        if self.processor == "HFApiInferenceImageGenProcessor":
            from src.processors.image.hf_img_gen_processor import HFApiInferenceImageGenProcessor
            self.client_session = aiohttp.ClientSession()
            return HFApiInferenceImageGenProcessor(
                aiohttp_session=self.client_session,
                api_key=os.environ.get("HF_API_KEY"),
                model="stabilityai/stable-diffusion-3.5-large",
            )
        if self.processor == "OpenAIImageGenProcessor":
            from src.processors.image.openai_img_gen_processor import OpenAIImageGenProcessor
            self.client_session = aiohttp.ClientSession()
            return OpenAIImageGenProcessor(
                image_size="1024x1024",
                aiohttp_session=self.client_session,
                model="stabilityai/stable-diffusion-3.5-large",
            )
        if self.processor == "TogetherImageGenProcessor":
            from src.processors.image.together_img_gen_processor import TogetherImageGenProcessor
            return TogetherImageGenProcessor(
                width=512,
                height=512,
                model="black-forest-labs/FLUX.1-schnell-Free",
            )

    async def asyncSetUp(self):
        pipeline = Pipeline([
            self.get_image_gen_processor(),
            FrameLogger(include_frame_types=[ImageRawFrame]),
            SaveImageProcessor(save_file=self.save_img_file),
        ])

        self.task = PipelineTask(
            pipeline,
            PipelineParams()
        )

    async def asyncTearDown(self):
        self.client_session and await self.client_session.close()
        pass

    async def test_run(self):
        runner = PipelineRunner()
        await self.task.queue_frames([
            TextFrame(
                text="A capybara holding a sign that reads Hello.",
            ),
        ])
        await self.task.stop_when_done()
        await runner.run(self.task)
