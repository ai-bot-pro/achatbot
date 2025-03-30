import os
import logging

from time import perf_counter
import unittest
import uuid
from dotenv import load_dotenv

from src.common.interface import ILlmGenerator
from src.core.llm.base import BaseLLM
from src.common.logger import Logger
from src.common.session import Session
from src.common.types import MODELS_DIR, SessionCtx
from src.core.llm import LLMEnvInit

load_dotenv(override=True)

r"""
# flashinfer needs to define TORCH_CUDA_ARCH_LIST
export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.7 8.9 9.0"

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

LLM_MODEL_NAME_OR_PATH=./models/Qwen/Qwen2.5-0.5B-Instruct-trtllm \
    LLM_TOKENIZER_PATH=/models/Qwen/Qwen2.5-0.5B-Instruct \
    LLM_DEBUG_MODE=1 \
    LLM_TAG=llm_trtllm_runner_generator \
    python -m unittest test.core.llm.test_generator.TestGenerator.test_generate
"""


class TestGenerator(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.llm_tag = os.getenv("LLM_TAG", "llm_transformers_generator")
        cls.prompt = os.getenv("LLM_PROMPT", "what's your name?")
        cls.model_path = os.getenv("LLM_MODEL_NAME_OR_PATH", "./models/Qwen/Qwen2.5-0.5B")
        cls.tokenizer_path = os.getenv("LLM_TOKENIZER_PATH", cls.model_path)
        Logger.init(os.getenv("LOG_LEVEL", "debug").upper(), is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    async def asyncSetUp(self):
        pass

    def setUp(self):
        from transformers import AutoTokenizer, GenerationConfig

        engine = LLMEnvInit.initLLMEngine(self.llm_tag)
        self.assertIsInstance(engine, BaseLLM)
        self.assertIsInstance(engine, ILlmGenerator)
        self.engine: ILlmGenerator = engine
        logging.debug(self.engine.args)

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.generation_config = {}
        if os.path.exists(os.path.join(self.model_path, "generation_config.json")):
            self.generation_config = GenerationConfig.from_pretrained(
                self.model_path, "generation_config.json"
            ).to_dict()

        self.session = Session(**SessionCtx(f"{self.llm_tag}_" + str(uuid.uuid4().hex)).__dict__)

    def tearDown(self):
        pass

    async def asyncTearDown(self):
        pass

    async def test_generate(self):
        prompt_cases = [
            {"prompt": "hello, my name is", "kwargs": {"max_new_tokens": 30, "stop_ids": []}},
            {
                "prompt": "hello, my name is",
                "kwargs": {"max_new_tokens": 30, "stop_ids": [13]},
            },  # prefill cache token test (trtllm default no cache, vllm and sglang default cache)
            {
                "prompt": "hello, what your name?",
                "kwargs": {"max_new_tokens": 30, "stop_ids": [13]},
            },
        ]

        for case in prompt_cases:
            with self.subTest(case=case):
                tokens = self.tokenizer(case["prompt"])
                self.session.ctx.state["token_ids"] = tokens["input_ids"]
                # gen_kwargs = {**self.generation_config, **case["kwargs"], **tokens}# hack test,
                # trtllm_runner, vllm  generate have some bug :)
                gen_kwargs = {**case["kwargs"], **tokens}
                logging.debug(gen_kwargs)
                iter = self.engine.generate(self.session, **gen_kwargs)

                generated_token_ids = []
                times = []
                start_time = perf_counter()
                async for item in iter:
                    # print(item)
                    generated_token_ids.append(item)
                    times.append(perf_counter() - start_time)
                    start_time = perf_counter()
                logging.debug(
                    f"generate TTFT time: {times[0]} s, {len(generated_token_ids)} tokens cost time: {sum(times)} s, avg cost time: {sum(times)/len(generated_token_ids)}"
                )
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
