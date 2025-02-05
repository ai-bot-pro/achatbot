import os
import logging

from time import perf_counter
import unittest
from dotenv import load_dotenv
from PIL import Image
import numpy as np

from src.core.llm.transformers.base import TransformersBaseLLM
from src.common.logger import Logger
from src.common.session import Session
from src.common.types import MODELS_DIR, TEST_DIR, SessionCtx
from src.core.llm import LLMEnvInit

load_dotenv(override=True)

r"""
LLM_DEVICE=cuda LLM_MODEL_NAME_OR_PATH=./models/deepseek-ai/Janus-Pro-1B \
    python -m unittest test.core.llm.test_transformers_img_janus.TestTransformersImgJanus.test_gen_imgs
"""


class TestTransformersImgJanus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompt = os.getenv("PROMPT", "超人大战钢铁侠")
        cls.llm_tag = os.getenv("LLM_TAG", "llm_transformers_manual_image_janus")
        Logger.init(os.getenv("LOG_LEVEL", "debug").upper(), is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.session = Session(**SessionCtx(f"test_{self.llm_tag}_client_id").__dict__)

        engine = LLMEnvInit.initLLMEngine(self.llm_tag)
        self.assertIsInstance(engine, TransformersBaseLLM)
        self.engine: TransformersBaseLLM = engine

    def tearDown(self):
        pass

    def test_gen_imgs(self):
        prompt_cases = [
            self.prompt,
        ]

        os.makedirs("generated_samples", exist_ok=True)
        i = 0
        for prompt in prompt_cases:
            print("\n--------test prompt: ", prompt, "--------\n")
            with self.subTest(prompt=prompt):
                self.session.ctx.state["prompt"] = prompt
                logging.debug(self.session.ctx)
                logging.debug(self.engine.args)
                iter = self.engine.generate(self.session)

                generated_text = ""
                times = []
                start_time = perf_counter()
                j = 0
                for item in iter:
                    save_path = os.path.join("generated_samples", f"pro_img_{i}_{j}.jpg")
                    Image.fromarray(np.frombuffer(item)).save(save_path)
                    times.append(perf_counter() - start_time)
                    start_time = perf_counter()
                    j += 1
                logging.debug(f"generate first image time: {times[0]} s")
                self.assertGreater(len(generated_text), 0)
                i += 1
