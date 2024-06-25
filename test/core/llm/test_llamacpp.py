import os
import logging

import unittest

from src.common.factory import EngineFactory
from src.core.llm.llamacpp import EngineClass, LLamacppLLM
from src.common.logger import Logger
from src.common.session import Session
from src.common.types import SessionCtx, MODELS_DIR


class TestLLamacppLLM(unittest.TestCase):
    r"""
    MODEL_TYPE=generate  python -m unittest test.core.llm.test_llamacpp.TestLLamacppLLM.test_generate
    MODEL_TYPE=generate STREAM=1 python -m unittest test.core.llm.test_llamacpp.TestLLamacppLLM.test_generate
    python -m unittest test.core.llm.test_llamacpp.TestLLamacppLLM.test_chat_completion
    STREAM=1 python -m unittest test.core.llm.test_llamacpp.TestLLamacppLLM.test_chat_completion
    """
    @classmethod
    def setUpClass(cls):
        cls.llm_tag = os.getenv('LLM_TAG', "llm_llamacpp")
        cls.prompt = os.getenv('PROMPT', "你好")
        cls.stream = os.getenv('STREAM', "")
        cls.model_type = os.getenv('MODEL_TYPE', "chat")
        cls.model_path = os.getenv('MODEL_PATH', os.path.join(
            MODELS_DIR, "qwen2-1_5b-instruct-q8_0.gguf"))
        Logger.init(logging.DEBUG)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        kwargs = {}
        kwargs["model_path"] = self.model_path
        kwargs["model_type"] = self.model_type
        kwargs["n_threads"] = os.cpu_count()
        self.llm: LLamacppLLM = EngineFactory.get_engine_by_tag(
            EngineClass, self.llm_tag, **kwargs)
        self.session = Session(**SessionCtx("test_client_id").__dict__)

    def tearDown(self):
        pass

    def test_generate(self):
        self.llm.args.llm_stream = bool(self.stream)
        logging.debug(self.llm.args)
        self.session.ctx.state["prompt"] = self.prompt
        logging.debug(self.session.ctx)
        iter = self.llm.generate(self.session)
        for item in iter:
            print(item)
            self.assertGreater(len(item), 0)

    def test_chat_completion(self):
        self.llm.args.llm_stream = bool(self.stream)
        self.llm.args.llm_stop = ["<|end|>", "</s>", "<s>", "<|user|>"]
        self.llm.args.llm_chat_system = "你是一个中国人,请用中文回答。回答限制在1-5句话内。要友好、乐于助人且简明扼要。保持对话简短而甜蜜。只用纯文本回答，不要包含链接或其他附加内容。不要回复计算机代码以及数学公式。"
        self.session.ctx.state["prompt"] = self.prompt
        logging.debug(self.session.ctx)
        iter = self.llm.chat_completion(self.session)
        for item in iter:
            print(item)
            self.assertGreater(len(item), 0)
