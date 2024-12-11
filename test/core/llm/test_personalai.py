import os
import logging

import unittest
from dotenv import load_dotenv

from src.core.llm.personalai import PersonalAIProxy
from src.common.logger import Logger
from src.common.session import Session
from src.common.types import SessionCtx
from src.core.llm import LLMEnvInit

load_dotenv(override=True)

r"""
API_URL=http://localhost:8787/ \
    python -m unittest test.core.llm.test_personalai.TestPersonalAIProxy.test_chat_completion
API_URL=http://localhost:8787/ \
    CHAT_TYPE=chat_with_functions \
    LLM_PROMPT=今天北京天气怎么样 \
    python -m unittest test.core.llm.test_personalai.TestPersonalAIProxy.test_chat_completion

API_URL=http://localhost:8787/ \
    CHAT_BOT=qianfan LLM_MODEL_NAME=completions \
    python -m unittest test.core.llm.test_personalai.TestPersonalAIProxy.test_chat_completion
API_URL=http://localhost:8787/ \
    CHAT_BOT=qianfan LLM_MODEL_NAME=completions \
    CHAT_TYPE=chat_with_functions \
    LLM_PROMPT=今天北京天气怎么样 \
    python -m unittest test.core.llm.test_personalai.TestPersonalAIProxy.test_chat_completion

API_URL=https://personal-ai-ts.weedge.workers.dev/ \
    CHAT_BOT=openai \
    python -m unittest test.core.llm.test_personalai.TestPersonalAIProxy.test_chat_completion
API_URL=https://personal-ai-ts.weedge.workers.dev/ \
    CHAT_BOT=openai \
    CHAT_TYPE=chat_with_functions \
    LLM_PROMPT=今天北京天气怎么样 \
    python -m unittest test.core.llm.test_personalai.TestPersonalAIProxy.test_chat_completion

API_URL=https://personal-ai-ts.weedge.workers.dev/ \
    CHAT_BOT=qianfan LLM_MODEL_NAME=completions \
    python -m unittest test.core.llm.test_personalai.TestPersonalAIProxy.test_chat_completion
API_URL=https://personal-ai-ts.weedge.workers.dev/ \
    CHAT_BOT=qianfan LLM_MODEL_NAME=completions\
    CHAT_TYPE=chat_with_functions \
    LLM_PROMPT=今天北京天气怎么样 \
    python -m unittest test.core.llm.test_personalai.TestPersonalAIProxy.test_chat_completion
"""


class TestPersonalAIProxy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llm_tag = os.getenv("LLM_TAG", "llm_personalai_proxy")
        cls.prompt = os.getenv("LLM_PROMPT", "你好")
        Logger.init(os.getenv("LOG_LEVEL", "debug").upper(), is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        kwargs = LLMEnvInit.map_config_func[self.llm_tag]()
        self.engine: PersonalAIProxy = PersonalAIProxy(**kwargs)
        logging.info(f"initLLMEngine: {self.llm_tag}, {self.engine}")
        self.session = Session(**SessionCtx(f"test_{self.llm_tag}_client_id").__dict__)

    def tearDown(self):
        pass

    def test_chat_completion(self):
        self.engine.args.llm_chat_system = "你是一个中国人,一名中文助理，请用中文简短回答，回答限制在1-5句话内。要友好、乐于助人且简明扼要。保持对话简短而甜蜜。只用纯文本回答，不要包含链接或其他附加内容。不要回复计算机代码以及数学公式。"
        self.session.ctx.state["prompt"] = self.prompt
        logging.debug(self.session.ctx)
        iter = self.engine.chat_completion(self.session)
        for item in iter:
            print(item)
            self.assertGreater(len(item), 0)
