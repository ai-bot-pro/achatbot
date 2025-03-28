import os
import logging

from time import perf_counter
import unittest
import uuid
from dotenv import load_dotenv

from src.core.llm.base import BaseLLM
from src.common.logger import Logger
from src.common.session import Session
from src.common.types import MODELS_DIR, SessionCtx
from src.core.llm import LLMEnvInit

load_dotenv(override=True)

r"""
LLM_MODEL_NAME_OR_PATH=./models/Qwen/Qwen2.5-0.5B-Instruct \
    LLM_TAG=llm_transformers_generator \
    python -m unittest test.core.llm.test_generator.TestGenerator.test_generate

LLM_MODEL_NAME_OR_PATH=./models/Qwen/Qwen2.5-0.5B-Instruct \
    LLM_MODEL_PATH=./models/qwen2.5-0.5b-instruct-q8_0.gguf \
    LLM_TAG=llm_llamacpp_generator \
    python -m unittest test.core.llm.test_generator.TestGenerator.test_generate

LLM_MODEL_NAME_OR_PATH=./models/Qwen/Qwen2.5-0.5B-Instruct \
    LLM_TAG=llm_vllm_generator \
    python -m unittest test.core.llm.test_generator.TestGenerator.test_generate

LLM_MODEL_NAME_OR_PATH=./models/Qwen/Qwen2.5-0.5B-Instruct \
    LLM_TAG=llm_sglang_generator \
    python -m unittest test.core.llm.test_generator.TestGenerator.test_generate

LLM_MODEL_NAME_OR_PATH=./models/Qwen/Qwen2.5-0.5B-Instruct \
    LLM_TAG=llm_trtllm_generator \
    python -m unittest test.core.llm.test_generator.TestGenerator.test_generate
"""


class TestGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llm_tag = os.getenv("LLM_TAG", "llm_transformers_generator")
        cls.prompt = os.getenv("LLM_PROMPT", "what's your name?")
        cls.model_path = os.getenv("LLM_MODEL_NAME_OR_PATH", "./models/Qwen/Qwen2.5-0.5B")
        Logger.init(os.getenv("LOG_LEVEL", "debug").upper(), is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        from transformers import AutoTokenizer, GenerationConfig

        engine = LLMEnvInit.initLLMEngine(self.llm_tag)
        self.assertIsInstance(engine, BaseLLM)
        self.engine: BaseLLM = engine
        logging.debug(self.engine.args)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.generation_config = {}
        if os.path.exists(os.path.join(self.model_path, "generation_config.json")):
            self.generation_config = GenerationConfig.from_pretrained(
                self.model_path, "generation_config.json"
            ).to_dict()

        self.session = Session(**SessionCtx(f"{self.llm_tag}_" + str(uuid.uuid4().hex)).__dict__)

    def tearDown(self):
        pass

    def test_generate(self):
        prompt_cases = [
            {"prompt": self.prompt, "kwargs": {"max_new_tokens": 20, "stop_ids": []}},
            {"prompt": self.prompt, "kwargs": {"max_new_tokens": 20, "stop_ids": [13]}},
        ]

        for case in prompt_cases:
            with self.subTest(case=case):
                tokens = self.tokenizer(case["prompt"])
                self.session.ctx.state["token_ids"] = tokens["input_ids"]
                gen_kwargs = {**self.generation_config, **case["kwargs"], **tokens}
                logging.debug(gen_kwargs)
                iter = self.engine.generate(self.session, **gen_kwargs)

                generated_token_ids = []
                times = []
                start_time = perf_counter()
                for item in iter:
                    # print(item)
                    generated_token_ids.append(item)
                    times.append(perf_counter() - start_time)
                    start_time = perf_counter()
                logging.debug(f"chat_completion TTFT time: {times[0]} s")
                self.assertGreater(len(generated_token_ids), 0)
                if "max_new_tokens" in case["kwargs"]:
                    max_new_tokens = case["kwargs"]["max_new_tokens"]
                    self.assertLessEqual(len(generated_token_ids), max_new_tokens)
                    if len(generated_token_ids) < max_new_tokens:
                        if "stop_ids" in case["kwargs"]:
                            for stop_id in case["kwargs"]["stop_ids"]:
                                self.assertIn(stop_id, generated_token_ids)
                generated_text = self.tokenizer.decode(generated_token_ids)
                logging.debug(f"generated text: {generated_text}")
