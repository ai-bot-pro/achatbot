import os
import logging
import uuid

import unittest
from PIL import Image

from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineTask, PipelineParams
from apipeline.pipeline.runner import PipelineRunner
from apipeline.frames.sys_frames import StopTaskFrame

from src.processors.vision.vision_processor import VisionProcessor
from src.common.session import Session
from src.common.types import SessionCtx, MODELS_DIR, TEST_DIR
from src.core.llm import LLMEnvInit
from src.common.logger import Logger
from apipeline.processors.logger import FrameLogger
from src.types.frames.data_frames import VisionImageRawFrame

from dotenv import load_dotenv

load_dotenv()


"""
LLM_MODEL_TYPE=chat LLM_MODEL_NAME="minicpm-v-2.6" \
    LLM_MODEL_PATH="./models/openbmb/MiniCPM-V-2_6-gguf/ggml-model-Q4_0.gguf" \
    LLM_CLIP_MODEL_PATH="./models/openbmb/MiniCPM-V-2_6-gguf/mmproj-model-f16.gguf" \
    LLM_CHAT_FORMAT=minicpm-v-2.6 \
    python -m unittest test.integration.processors.test_vision_processor.TestVisionProcessor

LLM_MODEL_TYPE=chat LLM_MODEL_NAME="minicpm-v-2.6" \
    LLM_MODEL_PATH="./models/openbmb/MiniCPM-V-2_6-gguf/ggml-model-Q4_0.gguf" \
    LLM_CLIP_MODEL_PATH="./models/openbmb/MiniCPM-V-2_6-gguf/mmproj-model-f16.gguf" \
    LLM_CHAT_FORMAT=minicpm-v-2.6 \
    python -m unittest test.integration.processors.test_vision_processor.TestVisionProcessor.test_run_text

LLM_TAG=llm_transformers_manual_vision_qwen \
    LLM_MODEL_NAME_OR_PATH="./models/Qwen/Qwen2-VL-2B-Instruct" \
    LLM_CHAT_HISTORY_SIZE=0 \
    python -m unittest test.integration.processors.test_vision_processor.TestVisionProcessor

LLM_TAG=llm_transformers_manual_vision_llama \
    LLM_MODEL_NAME_OR_PATH="./models/unsloth/Llama-3.2-11B-Vision-Instruct" \
    LLM_CHAT_HISTORY_SIZE=0 \
    python -m unittest test.integration.processors.test_vision_processor.TestVisionProcessor

LLM_TAG=llm_transformers_manual_vision_molmo \
    LLM_MODEL_NAME_OR_PATH=./models/allenai/Molmo-7B-D-0924 \
    LLM_CHAT_HISTORY_SIZE=0 \
    python -m unittest test.integration.processors.test_vision_processor.TestVisionProcessor
"""


class TestVisionProcessor(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)

        img_file = os.path.join(TEST_DIR, "img_files", "03-Confusing-Pictures.jpg")
        img_file = os.getenv("IMG_FILE", img_file)
        cls.img = Image.open(img_file)

    @classmethod
    def tearDownClass(cls):
        pass

    async def asyncSetUp(self):
        session = Session(**SessionCtx(uuid.uuid4()).__dict__)
        llm = LLMEnvInit.initLLMEngine()
        llm_processor = VisionProcessor(llm, session)
        pipeline = Pipeline(
            [
                FrameLogger(),
                llm_processor,
                FrameLogger(),
            ]
        )

        self.task = PipelineTask(pipeline, PipelineParams())

    async def asyncTearDown(self):
        pass

    async def test_run(self):
        runner = PipelineRunner()
        await self.task.queue_frames(
            [
                VisionImageRawFrame(
                    image=self.img.tobytes(),
                    size=self.img.size,
                    format=self.img.format,
                    mode=self.img.mode,
                    text="请描述下图片",
                ),
                StopTaskFrame(),
            ]
        )
        await runner.run(self.task)

    async def test_run_text(self):
        runner = PipelineRunner()
        await self.task.queue_frames(
            [
                VisionImageRawFrame(
                    text="你好",
                    image=bytes([]),
                    size=(0, 0),
                    format=None,
                    mode=None,
                ),
                StopTaskFrame(),
            ]
        )
        await runner.run(self.task)
