import os
import logging

from time import perf_counter
import unittest
from dotenv import load_dotenv

from src.common.interface import IVisionOCR
from src.modules.vision.ocr import VisionOCREnvInit
from src.common.logger import Logger
from src.common.session import Session
from src.common.types import SessionCtx

load_dotenv(override=True)

r"""
LLM_MODEL_NAME_OR_PATH=./models/stepfun-ai/GOT-OCR2_0 \
    python -m unittest test.modules.vision.ocr.test_transformers_got.TestTransformersGOTOCR.test_ocr_generate

"""


class TestTransformersGOTOCR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tag = os.getenv("VISION_OCR_TAG", "vision_transformers_got_ocr")
        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.session = Session(**SessionCtx(f"test_{self.tag}_client_id").__dict__)

        engine = VisionOCREnvInit.initVisionOCREngine(self.tag)
        self.assertIsInstance(engine, IVisionOCR)
        self.engine: IVisionOCR = engine

    def tearDown(self):
        pass

    def test_ocr_generate(self):
        image_cases = [
            {"image": "http://103.139.212.40/static/images/newprod/fpsb4/1.jpg"},
        ]

        for image in image_cases:
            with self.subTest(image=image):
                self.session.ctx.state["ocr_img"] = image
                iter = self.engine.stream_infer(self.session)

                generated_text = ""
                times = []
                start_time = perf_counter()
                for item in iter:
                    print(item, end="")
                    generated_text += item
                    times.append(perf_counter() - start_time)
                    start_time = perf_counter()
                self.assertGreater(len(times), 0)
                logging.info(f"generate TTFT time: {times[0]} s")
                logging.info(f"generated text: {generated_text}")
                self.assertGreater(len(generated_text), 0)
