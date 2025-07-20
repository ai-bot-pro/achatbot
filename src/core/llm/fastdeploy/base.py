from abc import abstractmethod
import logging
import os
from typing import AsyncGenerator

import numpy as np

from src.common.session import Session
from src.common.interface import ILlm
from src.common.chat_history import ChatHistory
from src.core.llm.base import BaseLLM
from src.common.logger import Logger


try:
    from fastdeploy.engine.args_utils import EngineArgs

    from src.types.llm.fastdeploy import FastDeployEngineArgs, LMGenerateArgs
    from src.core.llm.fastdeploy.generator import LLMEngineMonkey
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "you need to see https://paddlepaddle.github.io/FastDeploy/get_started/installation/nvidia_gpu/"
    )
    raise Exception(f"Missing module: {e}")


class FastdeployBase(BaseLLM, ILlm):
    def __init__(self, **kwargs) -> None:
        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)
        super().__init__()
        self.args = FastDeployEngineArgs(**kwargs)
        self.gen_args = LMGenerateArgs(**self.args.gen_args)
        # https://paddlepaddle.github.io/FastDeploy/parameters/
        self.serv_args = EngineArgs(**self.args.serv_args)
        logging.info(
            f"server args: {self.serv_args.__dict__} | default generate args: {self.gen_args.__dict__}"
        )
        self.engine = LLMEngineMonkey.from_engine_args(self.serv_args)

        if not self.engine.start():
            raise Exception("Failed to initialize FastDeploy LLM engine, service exit now!")
        logging.info(f"FastDeploy LLM engine initialized!")

        self.tokenizer = self.engine.data_processor.tokenizer

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

        self.warmup()

    def set_system_prompt(self, **kwargs):
        pass

    def init(self):
        pass

    @abstractmethod
    def warmup(self):
        raise NotImplementedError("must be implemented in the child class")

    def generate(self, session: Session, **kwargs):
        pass

    async def async_generate(
        self, session, **kwargs
    ) -> AsyncGenerator[str | dict | np.ndarray, None]:
        for item in self.generate(session, **kwargs):
            yield item

    async def async_chat_completion(self, session, **kwargs) -> AsyncGenerator[str, None]:
        logging.info("generate use chat_completion")
        for item in self.chat_completion(session):
            yield item

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
