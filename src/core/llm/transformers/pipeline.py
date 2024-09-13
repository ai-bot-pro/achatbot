import logging

from common.session import Session
from src.common.interface import ILlm
from src.core.llm.base import BaseLLM
from src.types.llm.transformers import TransformersLLMArgs
from src.types.speech.language import TO_LLM_LANGUAGE

class TransformersPipelineLLM(BaseLLM, ILlm):
    TAG = "llm_transformers_pipeline"

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**TransformersLLMArgs().__dict__, **kwargs}

    def __init__(self, **args) -> None:
        pass

    def generate(self, session: Session):
        # @TODO: personalai proxy need use comletions openai api
        logging.info(f"generate use chat_completion")
        for item in self.chat_completion(session):
            yield item

    def chat_completion(self, session: Session):
        if self.args.llm_stream is False:
            res = self._chat(session)
            yield res
        else:
            # yield from self._chat_stream(session)
            yield from self._chat(session)

    def count_tokens(self, text: str | bytes):
        pass

    def _chat(self, session: Session):
        pass

    def _chat_stream(self, session: Session):
        # !TODO: @weedge
        yield ""
