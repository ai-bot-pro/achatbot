import os
import logging
import uuid
from io import BytesIO

import unittest
from PIL import Image

from apipeline.pipeline.pipeline import Pipeline, FrameDirection
from apipeline.pipeline.task import PipelineTask, PipelineParams
from apipeline.pipeline.runner import PipelineRunner
from apipeline.frames.data_frames import Frame, TextFrame
from apipeline.frames.sys_frames import StopTaskFrame
from apipeline.processors.frame_processor import FrameProcessor

from src.processors.vision.llamacpp_v_processor import LLamaCPPVisionProcessor
from src.common.session import Session
from src.common.types import SessionCtx, MODELS_DIR, TEST_DIR
from src.core.llm import LLMEnvInit
from src.common.logger import Logger
from src.processors.frame_log_processor import FrameLogger
from src.types.frames.data_frames import VisionImageRawFrame

from dotenv import load_dotenv
load_dotenv()


"""
LLM_MODEL_TYPE=chat LLM_MODEL_NAME="minicpm-v-2.6" \
    LLM_MODEL_PATH="./models/openbmb/MiniCPM-V-2_6-gguf/ggml-model-Q4_0.gguf" \
    LLM_CLIP_MODEL_PATH="./models/openbmb/MiniCPM-V-2_6-gguf/mmproj-model-f16.gguf" \
    LLM_CHAT_FORMAT=minicpm-v-2.6 \
    python -m unittest test.integration.processors.test_llamacpp_v_processor.TestLLamaCPPVisionProcessor
"""


class TestLLamaCPPVisionProcessor(unittest.IsolatedAsyncioTestCase):

    @classmethod
    def setUpClass(cls):
        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)

        img_file = os.path.join(TEST_DIR, f"img_files", f"03-Confusing-Pictures.jpg")
        img_file = os.getenv('IMG_FILE', img_file)
        cls.img = Image.open(img_file)

    @classmethod
    def tearDownClass(cls):
        pass

    async def asyncSetUp(self):
        session = Session(**SessionCtx(uuid.uuid4()).__dict__)
        llm = LLMEnvInit.initLLMEngine()
        llm_processor = LLamaCPPVisionProcessor(llm, session)
        pipeline = Pipeline([
            FrameLogger(),
            llm_processor,
            FrameLogger(),
        ])

        self.task = PipelineTask(
            pipeline,
            PipelineParams()
        )

    async def asyncTearDown(self):
        pass

    async def test_run(self):
        runner = PipelineRunner()
        await self.task.queue_frames([
            VisionImageRawFrame(
                image=self.img.tobytes(),
                size=self.img.size,
                format=self.img.format,
                mode=self.img.mode,
                text="请描述下图片",
            ),
            StopTaskFrame(),
        ])
        await runner.run(self.task)
