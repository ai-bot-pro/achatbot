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

from src.processors.image import get_image_gen_processor
from src.common.logger import Logger
from apipeline.processors.logger import FrameLogger

from dotenv import load_dotenv

load_dotenv()


"""
mkdir -p images

IMAGE_GEN_PROCESSOR=HFApiInferenceImageGenProcessor \
    python -m unittest test.integration.processors.test_image_gen_processor.TestProcessor

IMAGE_GEN_PROCESSOR=OpenAIImageGenProcessor \
    python -m unittest test.integration.processors.test_image_gen_processor.TestProcessor

IMAGE_GEN_PROCESSOR=TogetherImageGenProcessor \
    python -m unittest test.integration.processors.test_image_gen_processor.TestProcessor

IMAGE_GEN_PROCESSOR=HFStableDiffusionImageGenProcessor \
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
            img = Image.frombytes(mode=frame.mode, size=frame.size, data=frame.image)
            img.save(self.save_file, frame.format)


class TestProcessor(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)
        cls.processor = os.getenv("IMAGE_GEN_PROCESSOR", "HFApiInferenceImageGenProcessor")
        cls.save_img_file = os.getenv("SAVE_FILE_PATH", "./images/test_gen.jpeg")
        cls.device = os.getenv("DEVICE", "auto")
        cls.client_session = None

    @classmethod
    def tearDownClass(cls):
        pass

    def get_image_gen_processor_by_tag(self):
        kwargs = {}
        if self.processor == "HFApiInferenceImageGenProcessor":
            self.client_session = aiohttp.ClientSession()
            kwargs["aiohttp_session"] = self.client_session
            kwargs["width"] = 1024
            kwargs["height"] = 1024
            kwargs["steps"] = 28
            kwargs["model"] = "stabilityai/stable-diffusion-3.5-large"
        if self.processor == "OpenAIImageGenProcessor":
            self.client_session = aiohttp.ClientSession()
            kwargs["aiohttp_session"] = self.client_session
            kwargs["image_size"] = "1024x1024"
            kwargs["model"] = "dall-e-3"
        if self.processor == "TogetherImageGenProcessor":
            kwargs["width"] = 1280
            kwargs["height"] = 720
            kwargs["steps"] = 4
            kwargs["model"] = "black-forest-labs/FLUX.1-schnell-Free"
        if self.processor == "HFStableDiffusionImageGenProcessor":
            kwargs["width"] = 1280
            kwargs["height"] = 720
            kwargs["steps"] = 28
            kwargs["guidance_scale"] = 3.5
            kwargs["is_quantizing"] = True
            kwargs["device"] = self.device
            kwargs["model"] = os.getenv("SD_MODEL", "stabilityai/stable-diffusion-3.5-large")

        return get_image_gen_processor(self.processor, **kwargs)

    def get_image_gen_processor(self):
        if self.processor == "HFApiInferenceImageGenProcessor":
            from src.processors.image.hf_img_gen_processor import HFApiInferenceImageGenProcessor

            self.client_session = aiohttp.ClientSession()
            return HFApiInferenceImageGenProcessor(
                aiohttp_session=self.client_session,
                model="stabilityai/stable-diffusion-3.5-large",
            )
        if self.processor == "OpenAIImageGenProcessor":
            from src.processors.image.openai_img_gen_processor import OpenAIImageGenProcessor

            self.client_session = aiohttp.ClientSession()
            return OpenAIImageGenProcessor(
                image_size="1024x1024",
                aiohttp_session=self.client_session,
                model="dall-e-3",
            )
        if self.processor == "TogetherImageGenProcessor":
            from src.processors.image.together_img_gen_processor import TogetherImageGenProcessor

            return TogetherImageGenProcessor(
                width=512,
                height=512,
                model="black-forest-labs/FLUX.1-schnell-Free",
            )

    async def asyncSetUp(self):
        pipeline = Pipeline(
            [
                self.get_image_gen_processor_by_tag(),
                FrameLogger(include_frame_types=[ImageRawFrame]),
                SaveImageProcessor(save_file=self.save_img_file),
            ]
        )

        self.task = PipelineTask(pipeline, PipelineParams())

    async def asyncTearDown(self):
        self.client_session and await self.client_session.close()
        pass

    async def test_run(self):
        runner = PipelineRunner()
        await self.task.queue_frames(
            [
                TextFrame(
                    text="A capybara holding a sign that reads Hello.",
                ),
            ]
        )
        await self.task.stop_when_done()
        await runner.run(self.task)
