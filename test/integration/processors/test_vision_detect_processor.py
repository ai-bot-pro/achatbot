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

from src.processors.vision.detect_processor import DetectProcessor
from src.common.factory import EngineClass
from src.common.interface import IVisionDetector
from src.modules.vision.detector import VisionDetectorEnvInit
from src.common.session import Session
from src.common.types import SessionCtx, MODELS_DIR, TEST_DIR
from src.common.logger import Logger
from src.types.frames.data_frames import UserImageRawFrame
from src.types.vision.detector.yolo import CustomConfidence, VisionDetectorArgs

from dotenv import load_dotenv

load_dotenv()


"""
python -m unittest test.integration.processors.test_vision_detect_processor.TestProcessor.test_run

python -m unittest test.integration.processors.test_vision_detect_processor.TestProcessor.test_run_empty
"""


class TestProcessor(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)

        img_file = os.path.join(TEST_DIR, "img_files", "dog.jpeg")
        img_file = os.getenv("IMG_FILE", img_file)
        cls.img = Image.open(img_file)
        cls.tag = os.getenv("DETECTOR_TAG", "vision_yolo_detector")
        cls.model = os.getenv("YOLO_MODEL", os.path.join(MODELS_DIR, "yolov10n.pt"))
        cls.stream = bool(os.getenv("YOLO_STREAM", "0"))
        cls.classes = os.getenv("YOLO_WD_CLASSES", "person,backpack,dog,eye,nose,ear,tongue").split(
            ","
        )
        cls.detected_text = os.getenv("DETECTED_TEXT", "你好。")
        cls.out_detected_text = os.getenv("OUT_DETECTED_TEXT", "拜拜。")

    @classmethod
    def tearDownClass(cls):
        pass

    def get_vision_processor(self):
        kwargs = VisionDetectorArgs(
            model=self.model,
            verbose=False,
            stream=self.stream,
            custom_classes=self.classes,
            # conf=0.0,  # "Confidence Threshold", 0.0, 1.0, 0.25, 0.01
            # iou=0.0,  # "IoU Threshold", 0.0, 1.0, 0.45, 0.01
            selected_classes=[],
            custom_confidences=[
                CustomConfidence(
                    boolean_op="and", class_name="person", d_confidence=0.3, d_min_cn=1
                ),
                CustomConfidence(boolean_op="and", class_name="dog", d_confidence=0.5, d_min_cn=1),
                CustomConfidence(boolean_op="or", class_name="car", d_confidence=0.3, d_min_cn=1),
            ],
        ).__dict__
        print(kwargs)
        self.detector: IVisionDetector | EngineClass = VisionDetectorEnvInit.getEngine(
            self.tag, **kwargs
        )
        self.session = Session(**SessionCtx(f"{__class__}").__dict__)
        return DetectProcessor(
            detected_text=self.detected_text,
            out_detected_text=self.out_detected_text,
            detector=self.detector,
            session=self.session,
        )

    async def out_cb(self, frame):
        # await asyncio.sleep(1)
        logging.info(f"sink_callback print frame: {frame}")
        self.assertIsInstance(frame, TextFrame)
        self.assertEqual(frame.text, self.detected_text)

    async def asyncSetUp(self):
        pipeline = Pipeline(
            [
                self.get_vision_processor(),
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
