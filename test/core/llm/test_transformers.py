import os
import logging

from time import perf_counter
import unittest
from dotenv import load_dotenv

from src.core.llm.transformers.base import TransformersBaseLLM
from src.common.logger import Logger
from src.common.session import Session
from src.common.types import SessionCtx
from src.core.llm import LLMEnvInit

load_dotenv(override=True)

r"""
LLM_TAG=llm_transformers_manual \
    python -m unittest test.core.llm.test_transformers.TestTransformers.test_chat_completion
LLM_TAG=llm_transformers_manual \
    python -m unittest test.core.llm.test_transformers.TestTransformers.test_chat_completion_zh
LLM_TAG=llm_transformers_manual \
    python -m unittest test.core.llm.test_transformers.TestTransformers.test_chat_completion_prompts

LLM_TAG=llm_transformers_pipeline \
    python -m unittest test.core.llm.test_transformers.TestTransformers.test_chat_completion
LLM_TAG=llm_transformers_pipeline \
    python -m unittest test.core.llm.test_transformers.TestTransformers.test_chat_completion_zh
LLM_TAG=llm_transformers_pipeline \
    python -m unittest test.core.llm.test_transformers.TestTransformers.test_chat_completion_prompts

"""


class TestTransformers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llm_tag = os.getenv("LLM_TAG", "llm_transformers_manual")
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

    def test_chat_completion(self):
        self.session.ctx.state["prompt"] = self.prompt
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

    def test_chat_completion_zh(self):
        self.engine.args.init_chat_prompt = "你叫马里奥，一名中文助理，请用中文简短回答，回答限制在1-5句话内。要友好、乐于助人且简明扼要。保持对话简短而甜蜜。只用纯文本回答，不要包含链接或其他附加内容。不要回复计算机代码以及数学公式。"
        self.engine.chat_history.init(
            {
                "role": self.engine.args.init_chat_role,
                "content": self.engine.args.init_chat_prompt,
            }
        )
        self.session.ctx.state["prompt"] = "你叫什么名字？"
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

    def test_chat_completion_prompts(self):
        prompt_cases = [
            self.prompt,
            (self.prompt, "en"),
            (self.prompt, "zh"),
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
