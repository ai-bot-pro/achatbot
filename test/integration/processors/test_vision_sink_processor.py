import asyncio
import os
import logging
import uuid

import unittest
from PIL import Image

from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineTask, PipelineParams
from apipeline.pipeline.runner import PipelineRunner
from apipeline.frames.sys_frames import StopTaskFrame
from apipeline.frames.data_frames import TextFrame
from apipeline.frames.control_frames import EndFrame
from apipeline.processors.output_processor import OutputFrameProcessor
from apipeline.processors.aggregators.sentence import SentenceAggregator

from src.processors.aggregators.vision_image_frame import VisionImageFrameAggregator
from src.processors.user_image_request_processor import UserImageRequestProcessor
from src.processors.vision.vision_processor import MockVisionProcessor, VisionProcessor
from src.common.session import Session
from src.common.types import SessionCtx, MODELS_DIR, TEST_DIR
from src.core.llm import LLMEnvInit
from src.common.logger import Logger
from apipeline.processors.logger import FrameLogger
from src.types.frames.data_frames import VisionImageRawFrame

from dotenv import load_dotenv

load_dotenv()


"""
VISION_PROCESSOR_TAG=mock_vision_processor \
    python -m unittest test.integration.processors.test_vision_sink_processor.TestVisionProcessor.test_run

VISION_PROCESSOR_TAG=vision_processor \
    LLM_TAG=llm_transformers_manual_vision_qwen \
    LLM_MODEL_NAME_OR_PATH="./models/Qwen/Qwen2-VL-2B-Instruct" \
    LLM_CHAT_HISTORY_SIZE=0 \
    python -m unittest test.integration.processors.test_vision_sink_processor.TestVisionProcessor
"""


class TestVisionProcessor(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)

        img_file = os.path.join(TEST_DIR, "img_files", "03-Confusing-Pictures.jpg")
        img_file = os.getenv("IMG_FILE", img_file)
        cls.img = Image.open(img_file)
        cls.vision_tag = os.getenv("VISION_PROCESSOR_TAG", "mock_vision_processor")
        cls.mock_text = os.getenv("MOCK_TEXT", "你他娘的是个人才。")

    @classmethod
    def tearDownClass(cls):
        pass

    def get_vision_llm_processor(self):
        if self.vision_tag == "mock_vision_processor":
            return MockVisionProcessor(self.mock_text)

        session = Session(**SessionCtx(uuid.uuid4()).__dict__)
        llm = LLMEnvInit.initLLMEngine()
        return VisionProcessor(llm=llm, session=session)

    async def out_cb(self, frame):
        await asyncio.sleep(1)
        logging.info(f"sink_callback print frame: {frame}")
        if self.vision_tag == "mock_vision_processor":
            if isinstance(frame, TextFrame):
                self.assertEqual(frame.text, self.mock_text)

    async def asyncSetUp(self):
        pipeline = Pipeline(
            [
                # SentenceAggregator(),
                # UserImageRequestProcessor(),
                # VisionImageFrameAggregator(),
                self.get_vision_llm_processor(),
                OutputFrameProcessor(cb=self.out_cb),
            ]
        )
        self.vision_task = PipelineTask(
            pipeline,
            params=PipelineParams(),
        )

    async def asyncTearDown(self):
        pass

    async def test_run(self):
        runner = PipelineRunner()
        await self.vision_task.queue_frames(
            [
                VisionImageRawFrame(
                    image=self.img.tobytes(),
                    size=self.img.size,
                    format=self.img.format,
                    mode=self.img.mode,
                    text="请描述下图片",
                ),
                EndFrame(),
            ]
        )
        await runner.run(self.vision_task)

    async def test_run_text(self):
        runner = PipelineRunner()
        await self.vision_task.queue_frames(
            [
                VisionImageRawFrame(
                    text="你好",
                    image=bytes([]),
                    size=(0, 0),
                    format=None,
                    mode=None,
                ),
                EndFrame(),
            ]
        )
        await runner.run(self.vision_task)
