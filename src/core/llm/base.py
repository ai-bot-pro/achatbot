import re

from src.common.interface import ILlm
from src.common.factory import EngineClass


class BaseLLM(EngineClass, ILlm):
    def model_name(self):
        if hasattr(self.args, "model_name"):
            return self.args.model_name
        return ""

    def _have_special_char(self, content: str, use_nltk: bool = False) -> int:
        """
        check content to match a specail char which is end of sentence
        !TODO: use NLTK sent_tokenize to get sentences. @weedge
        """
        if use_nltk is False:
            pattern = r"""[.。,，;；!！?？、]"""
            matches = re.findall(pattern, content)
            if len(matches) == 0:
                return -1
            return content.index(matches[len(matches) - 1])

    def generate(self, session, **kwargs):
        """
        generate text or tokens with stream iterator
        - local llm cpu/gpu bind
        - api llm io bind
        """
        pass

    def chat_completion(self, session, **kwargs):
        pass

    def count_tokens(self, text: str | bytes):
        pass
