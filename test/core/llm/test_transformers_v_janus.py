import os
import logging

from time import perf_counter
import unittest
from dotenv import load_dotenv
from PIL import Image

from src.core.llm.transformers.base import TransformersBaseLLM
from src.common.logger import Logger
from src.common.session import Session
from src.common.types import MODELS_DIR, TEST_DIR, SessionCtx
from src.core.llm import LLMEnvInit

load_dotenv(override=True)

r"""
LLM_DEVICE=cuda LLM_MODEL_NAME_OR_PATH=./models/deepseek-ai/Janus-Pro-1B \
    python -m unittest test.core.llm.test_transformers_v_janus.TestTransformersVJanus.test_chat_completion_prompts

LLM_DEVICE=cuda LLM_TAG=llm_transformers_manual_vision_janus_flow \
    LLM_MODEL_NAME_OR_PATH=./models/deepseek-ai/JanusFlow-1.3B \
    python -m unittest test.core.llm.test_transformers_v_janus.TestTransformersVJanus.test_chat_completion_prompts
"""


class TestTransformersVJanus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        img_file = os.path.join(TEST_DIR, "img_files", "03-Confusing-Pictures.jpg")
        cls.img_file = os.getenv("IMG_FILE", img_file)
        cls.prompt = os.getenv("PROMPT", "描述下图片中的内容")
        cls.llm_tag = os.getenv("LLM_TAG", "llm_transformers_manual_vision_janus")
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

    def test_chat_completion_prompts(self):
        prompt_cases = [
            [Image.open(self.img_file), self.prompt],
        ]

        for prompt in prompt_cases:
            print("\n--------test prompt: ", prompt, "--------\n")
            with self.subTest(prompt=prompt):
                self.session.ctx.state["prompt"] = prompt
                logging.debug(self.session.ctx)
                logging.debug(self.engine.args)
                iter = self.engine.chat_completion(self.session)

                generated_text = ""
                times = []
                start_time = perf_counter()
                for item in iter:
                    # print(item)
                    generated_text += item
                    times.append(perf_counter() - start_time)
                    start_time = perf_counter()
                logging.debug(f"chat_completion TTFT time: {times[0]} s")
                logging.debug(f"generated text: {generated_text}")
                self.assertGreater(len(generated_text), 0)
