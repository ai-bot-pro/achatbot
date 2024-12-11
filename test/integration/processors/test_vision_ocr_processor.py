import asyncio
import os
import logging
import uuid

import unittest
from PIL import Image

from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineTask, PipelineParams
from apipeline.pipeline.runner import PipelineRunner
from apipeline.frames.data_frames import TextFrame
from apipeline.frames.control_frames import EndFrame
from apipeline.processors.output_processor import OutputFrameProcessor
from apipeline.processors.logger import FrameLogger

from src.modules.vision.ocr import VisionOCREnvInit
from src.processors.vision.ocr_processor import OCRProcessor
from src.common.factory import EngineClass
from src.common.interface import IVisionDetector, IVisionOCR
from src.common.session import Session
from src.common.types import SessionCtx, MODELS_DIR, TEST_DIR
from src.common.logger import Logger
from src.types.frames.data_frames import UserImageRawFrame
from src.types.vision.detector.yolo import CustomConfidence, VisionDetectorArgs

from dotenv import load_dotenv

load_dotenv()


"""
python -m unittest test.integration.processors.test_vision_ocr_processor.TestProcessor.test_run

python -m unittest test.integration.processors.test_vision_ocr_processor.TestProcessor.test_run_empty
"""


class TestProcessor(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)

        img_file = os.path.join(TEST_DIR, "img_files", "ocr.jpg")
        img_file = os.getenv("IMG_FILE", img_file)
        cls.img = Image.open(img_file)
        cls.tag = os.getenv("VISION_OCR_TAG", "vision_transformers_got_ocr")

    @classmethod
    def tearDownClass(cls):
        pass

    def get_vision_ocr_processor(self):
        self.session = Session(**SessionCtx(f"test_{self.tag}_client_id").__dict__)
        engine: IVisionOCR = VisionOCREnvInit.initVisionOCREngine(self.tag)
        return OCRProcessor(ocr=engine, session=self.session)

    async def out_cb(self, frame):
        # await asyncio.sleep(1)
        logging.info(f"sink_callback print frame: {frame}")
        self.assertIsInstance(frame, TextFrame)

    async def asyncSetUp(self):
        pipeline = Pipeline(
            [
                self.get_vision_ocr_processor(),
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
                UserImageRawFrame(
                    image=self.img.tobytes(),
                    size=self.img.size,
                    format=self.img.format,
                    mode=self.img.mode,
                    user_id="110",
                ),
                EndFrame(),
            ]
        )
        await runner.run(self.vision_task)

    async def test_run_empty(self):
        runner = PipelineRunner()
        await self.vision_task.queue_frames(
            [
                UserImageRawFrame(
                    image=bytes([]),
                    size=(0, 0),
                    format=None,
                    mode=None,
                    user_id="110",
                ),
                EndFrame(),
            ]
        )
        await runner.run(self.vision_task)
