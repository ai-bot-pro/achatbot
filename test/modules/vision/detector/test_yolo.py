import os
import logging

import unittest
from PIL import Image

from src.modules.vision.detector import VisionDetectorEnvInit
from src.common.logger import Logger
from src.common.factory import EngineClass
from src.common.session import Session
from src.common.types import SessionCtx, TEST_DIR, MODELS_DIR
from src.common.interface import IVisionDetector
from src.types.vision.detector.yolo import CustomConfidence, VisionDetectorArgs

r"""
python -m unittest test.modules.vision.detector.test_yolo.TestYOLODetector.test_detect
YOLO_MODEL=./models/yolov10n.pt python -m unittest test.modules.vision.detector.test_yolo.TestYOLODetector.test_detect
YOLO_MODEL=./models/yolov8s-worldv2.pt python -m unittest test.modules.vision.detector.test_yolo.TestYOLODetector.test_detect

python -m unittest test.modules.vision.detector.test_yolo.TestYOLODetector.test_annotate
YOLO_MODEL=./models/yolov10n.pt python -m unittest test.modules.vision.detector.test_yolo.TestYOLODetector.test_annotate
YOLO_MODEL=./models/yolov8s-worldv2.pt python -m unittest test.modules.vision.detector.test_yolo.TestYOLODetector.test_annotate
"""


class TestYOLODetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tag = os.getenv("DETECTOR_TAG", "vision_yolo_detector")
        img_file = os.path.join(TEST_DIR, "img_files", "dog.jpeg")
        cls.img_file = os.getenv("IMG_FILE", img_file)
        cls.model = os.getenv("YOLO_MODEL", os.path.join(MODELS_DIR, "yolov8n.pt"))
        cls.stream = bool(os.getenv("YOLO_STREAM", "0"))
        cls.classes = os.getenv("YOLO_WD_CLASSES", "person,backpack,dog,eye,nose,ear,tongue").split(
            ","
        )

        Logger.init(os.getenv("LOG_LEVEL", "debug").upper(), is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        kwargs = VisionDetectorArgs(
            model=self.model,
            verbose=True,
            stream=self.stream,
            custom_classes=self.classes,
            custom_confidences=[
                CustomConfidence(
                    boolean_op="and", class_name="person", d_confidence=0.3, d_min_cn=1
                ),
                CustomConfidence(boolean_op="and", class_name="dog", d_confidence=0.5, d_min_cn=1),
                CustomConfidence(boolean_op="or", class_name="car", d_confidence=0.2, d_min_cn=1),
            ],
        ).__dict__
        print(kwargs)
        self.detector: IVisionDetector | EngineClass = VisionDetectorEnvInit.getEngine(
            self.tag, **kwargs
        )
        self.session = Session(**SessionCtx(f"{__class__}").__dict__)

    def tearDown(self):
        pass

    def test_detect(self):
        img_file = os.path.join(TEST_DIR, "img_files", "dog.jpeg")
        self.session.ctx.state["detect_img"] = Image.open(img_file)
        is_detected = self.detector.detect(self.session)
        self.assertEqual(is_detected, True)

    def test_annotate(self):
        img_file = os.path.join(TEST_DIR, "img_files", "dog.jpeg")
        self.session.ctx.state["detect_img"] = Image.open(img_file)
        img_iter = self.detector.annotate(self.session)
        for img in img_iter:
            print(img)
            self.assertIsInstance(img, Image.Image)
