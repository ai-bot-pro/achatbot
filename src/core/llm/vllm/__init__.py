from typing import AsyncGenerator, Iterator

import numpy as np

from src.common.session import Session
from src.common.interface import ILlm
from src.core.llm.base import BaseLLM


class VLlmBase(BaseLLM, ILlm):
    def set_system_prompt(self, **kwargs):
        pass

    def generate(self, session: Session, **kwargs) -> Iterator[str | dict | np.ndarray]:
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
