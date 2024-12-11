import os
import logging

from time import perf_counter
import unittest
from dotenv import load_dotenv

from src.core.llm.transformers.base import TransformersBaseLLM
from src.common.logger import Logger
from src.common.session import Session
from src.common.types import MODELS_DIR, SessionCtx
from src.core.llm import LLMEnvInit

load_dotenv(override=True)

r"""
LLM_MODEL_NAME_OR_PATH=./models/unsloth/Llama-3.2-11B-Vision-Instruct \
    python -m unittest test.core.llm.test_transformers_v_llama.TestTransformersVLlama.test_chat_completion_prompts

LLM_TAG=llm_transformers_manual_vision_molmo \
    LLM_MODEL_NAME_OR_PATH=./models/allenai/Molmo-7B-D-0924 \
    python -m unittest test.core.llm.test_transformers_v_llama.TestTransformersVLlama.test_chat_completion_prompts
"""


class TestTransformersVLlama(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llm_tag = os.getenv("LLM_TAG", "llm_transformers_manual_vision_llama")
        cls.prompt = os.getenv("LLM_PROMPT", "what's your name?")
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
            self.prompt,
            (self.prompt, "en"),
            (self.prompt, "zh"),
            [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {
                    "type": "text",
                    "text": "Describe this video. Please reply to my message in chinese",
                },
            ],
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
