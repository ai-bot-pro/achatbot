from abc import abstractmethod
import logging
import os
from queue import Queue
from typing import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
import uuid

import numpy as np
from PIL import Image

try:
    from transformers import AutoTokenizer
    from vllm import LLM, AsyncLLMEngine, AsyncEngineArgs, SamplingParams
    from src.types.llm.vllm import VllmEngineArgs, LMGenerateArgs
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error("you need to `pip install achatbot[vllm]`")
    raise Exception(f"Missing module: {e}")

from src.common.session import Session
from src.common.interface import ILlm
from src.common.chat_history import ChatHistory
from src.core.llm.base import BaseLLM
from src.common.logger import Logger
from src.common.utils import task
from src.common.types import SessionCtx


Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)


class VllmBase(BaseLLM, ILlm):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.args = VllmEngineArgs(**kwargs)
        self.gen_args = LMGenerateArgs(**self.args.gen_args)
        # https://docs.vllm.ai/en/stable/api/vllm/engine/arg_utils.html#vllm.engine.arg_utils.AsyncEngineArgs.__init__
        self.serv_args = AsyncEngineArgs(**self.args.serv_args)
        logging.info(
            f"server args: {self.serv_args.__dict__} | default generate args: {self.gen_args.__dict__}"
        )
        self.engine = AsyncLLMEngine.from_engine_args(self.serv_args)
        self.tokenizer = AutoTokenizer.from_pretrained(self.serv_args.model)

        self._chat_history = ChatHistory(self.args.chat_history_size)
        if self.args.init_chat_role and self.args.init_chat_prompt:
            self._chat_history.init(
                {
                    "role": self.args.init_chat_role,
                    "content": self.args.init_chat_prompt,
                }
            )

        # subclass to init
        self.init()

    def set_system_prompt(self, **kwargs):
        pass

    def init(self):
        pass

    async def warmup(self):
        if self.args.warmup_steps <= 0:
            return
        session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)

        dummy_pil_image = Image.new("RGB", (100, 100), color="white")
        session.ctx.state["prompt"] = [
            {"type": "image", "image": dummy_pil_image},
            {"type": "text", "text": self.args.warmup_prompt},
        ]
        for i in range(self.args.warmup_steps):
            logging.info(f"{i} warmup start")
            async for result_text in self.async_generate(
                session, thinking=self.gen_args.lm_gen_thinking
            ):
                pass

    def generate(self, session: Session, **kwargs):
        pass

    async def async_generate(
        self, session, **kwargs
    ) -> AsyncGenerator[str | dict | np.ndarray, None]:
        pass

    async def async_chat_completion(self, session, **kwargs) -> AsyncGenerator[str, None]:
        if self.args.lm_stream is False:
            res = ""
            async for text in self.async_generate(session, **kwargs):
                res += text
            yield res
        else:
            res = ""
            async for text in self.async_generate(session, **kwargs):
                if text is None:
                    yield None
                    continue
                res += text
                pos = self._have_special_char(res)
                if pos > -1:
                    yield res[: pos + 1]
                    res = res[pos + 1 :]
                else:
                    yield None
            if len(res) > 0:
                yield res

    def chat_completion(self, session: Session, **kwargs):
        if self.args.lm_stream is False:
            res = ""
            for text in self.generate(session, **kwargs):
                res += text
            yield res
        else:
            res = ""
            for text in self.generate(session, **kwargs):
                if text is None:
                    yield None
                    continue
                res += text
                pos = self._have_special_char(res)
                if pos > -1:
                    yield res[: pos + 1]
                    res = res[pos + 1 :]
                else:
                    yield None
            if len(res) > 0:
                yield res

    def count_tokens(self, text: str | bytes) -> int:
        """
        use sentencepiece tokenizer to count tokens
        """
        return len(self.tokenizer.tokenize(text)) if self.tokenizer else 0
