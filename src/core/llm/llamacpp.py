import logging
import re

from src.common.interface import ILlm
from .base import BaseLLM
from src.common.session import Session
from src.common.types import LLamcppLLMArgs


class LLamacppLLM(BaseLLM, ILlm):
    TAG = "llm_llamacpp"

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**LLamcppLLMArgs().__dict__, **kwargs}

    def __init__(self, **args) -> None:
        self.args = LLamcppLLMArgs(**args)
        from llama_cpp import Llama
        if "chat" in self.args.model_type.lower():
            self.model = Llama(
                model_path=self.args.model_path,
                n_ctx=self.args.n_ctx,
                verbose=self.args.verbose,
                n_batch=self.args.n_batch,
                n_threads=self.args.n_threads,
                n_gpu_layers=self.args.n_gpu_layers,
                flash_attn=self.args.flash_attn,
                chat_format=self.args.chat_format)
        else:
            self.model = Llama(
                model_path=self.args.model_path,
                n_ctx=self.args.n_ctx,
                verbose=self.args.verbose,
                n_batch=self.args.n_batch,
                n_threads=self.args.n_threads,
                flash_attn=self.args.flash_attn,
                n_gpu_layers=self.args.n_gpu_layers)

    def encode(self, text: str | bytes):
        return self.model.tokenize(text.encode() if isinstance(text, str) else text)

    def count_tokens(self, text: str | bytes):
        return len(self.encode(text))

    def generate(self, session: Session):
        prompt = session.ctx.state["prompt"]
        prompt = self.args.llm_prompt_tpl % (prompt,)
        output = self.model(
            prompt,
            max_tokens=self.args.llm_max_tokens,  # Generate up to 256 tokens
            stop=self.args.llm_stop,
            # echo=True,  # Whether to echo the prompt
            stream=self.args.llm_stream,
            temperature=self.args.llm_temperature,
            top_p=self.args.llm_top_p,
            top_k=self.args.llm_top_k,
        )
        logging.debug(f"llm generate: {output}")
        res = ""
        if self.args.llm_stream:
            for item in output:
                content = item['choices'][0]['text']
                res += content
                pos = self._have_special_char(res)
                if pos > -1:
                    yield res[:pos + 1]
                    res = res[pos + 1:]
            if len(res) > 0:
                yield res
        else:
            yield output['choices'][0]['text']

    def chat_completion(self, session: Session):
        query = session.ctx.state["prompt"]
        output = self.model.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": self.args.llm_chat_system,
                },
                {"role": "user", "content": query},
            ],
            # response_format={"type": "json_object"},
            max_tokens=self.args.llm_max_tokens,
            stop=self.args.llm_stop,
            stream=self.args.llm_stream,
            temperature=self.args.llm_temperature,
            top_p=self.args.llm_top_p,
            top_k=self.args.llm_top_k,
            tool_choice=self.args.llm_tool_choice,
            tools=self.args.llm_tools,
        )
        res = ""
        if self.args.llm_stream:
            for item in output:
                if 'content' in item['choices'][0]['delta']:
                    content = item['choices'][0]['delta']['content']
                    res += content
                    pos = self._have_special_char(res)
                    if pos > -1:
                        yield res[:pos + 1]
                        res = res[pos + 1:]
            if len(res) > 0:
                yield res
        else:
            res = output['choices'][0]['message']['content'] if 'content' in output['choices'][0]['message'] else ""
            yield res

    def _have_special_char(self, content: str) -> int:
        pattern = r"""[.。,，;；!！?？、]"""
        matches = re.findall(pattern, content)
        if len(matches) == 0:
            return -1
        return content.index(matches[len(matches) - 1])
