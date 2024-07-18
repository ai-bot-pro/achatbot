import logging
import json
import re

from src.common.interface import ILlm
from .base import BaseLLM
from src.common.session import Session
from src.common.types import LLamcppLLMArgs
from src.modules.functions.function import FunctionManager


class LLamacppLLM(BaseLLM, ILlm):
    TAG = "llm_llamacpp"

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**LLamcppLLMArgs().__dict__, **kwargs}

    def __init__(self, **args) -> None:
        self.args = LLamcppLLMArgs(**args)
        from llama_cpp import Llama
        from llama_cpp.llama_tokenizer import LlamaHFTokenizer

        if self.args.tokenizer_path is not None:
            self.model = Llama(
                model_path=self.args.model_path,
                n_ctx=self.args.n_ctx,
                verbose=self.args.verbose,
                n_batch=self.args.n_batch,
                n_threads=self.args.n_threads,
                n_gpu_layers=self.args.n_gpu_layers,
                flash_attn=self.args.flash_attn,
                chat_format=self.args.chat_format,
                tokenizer=LlamaHFTokenizer.from_pretrained(self.args.tokenizer_path),
            )
        else:
            self.model = Llama(
                model_path=self.args.model_path,
                n_ctx=self.args.n_ctx,
                verbose=self.args.verbose,
                n_batch=self.args.n_batch,
                n_threads=self.args.n_threads,
                n_gpu_layers=self.args.n_gpu_layers,
                flash_attn=self.args.flash_attn,
                chat_format=self.args.chat_format,
            )

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
        if self.args.model_type not in ["chat", "chat-func"]:
            yield from self.generate(session)
            return
        query = session.ctx.state["prompt"]
        session.chat_history.append({"role": "user", "content": query})
        res = ""
        if self.args.model_type == "chat":
            for item in self._chat_completion(session):
                res += item
                yield item
        elif self.args.model_type == "chat-func":
            for item in self._chat_completion_functions(session):
                res += item
                yield item
        session.chat_history.append({"role": "assistant", "content": res})
        logging.debug(f"chat_history:{session.chat_history}")

    def _chat_completion(self, session: Session):
        output = self.model.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": self.args.llm_chat_system,
                },
                *session.chat_history
            ],
            # response_format={"type": "json_object"},
            max_tokens=self.args.llm_max_tokens,
            stop=self.args.llm_stop,
            stream=self.args.llm_stream,
            temperature=self.args.llm_temperature,
            top_p=self.args.llm_top_p,
            top_k=self.args.llm_top_k,
        )
        res = ""
        if self.args.llm_stream:
            for item in output:
                if 'content' in item['choices'][0]['delta']:
                    content = item['choices'][0]['delta']['content']
                    if content is None:
                        continue
                    res += content
                    pos = self._have_special_char(res)
                    if pos > -1:
                        yield res[:pos + 1]
                        res = res[pos + 1:]
            if len(res) > 0:
                yield res
        else:
            res = output['choices'][0]['message']['content'] if 'content' in output['choices'][0]['message'] else ""
            if res is not None:
                yield res

    def _chat_completion_functions(self, session: Session):
        tools = FunctionManager.get_tool_calls()
        logging.debug(f"tools: {tools}")
        while True:
            messages = [
                {
                    "role": "system",
                    "content": self.args.llm_chat_system,
                },
                *session.chat_history,
            ]
            logging.debug(f"messages: {messages}")
            output = self.model.create_chat_completion(
                messages=messages,
                max_tokens=self.args.llm_max_tokens,
                stop=self.args.llm_stop,
                stream=self.args.llm_stream,
                temperature=self.args.llm_temperature,
                top_p=self.args.llm_top_p,
                top_k=self.args.llm_top_k,
                tools=tools,
                tool_choice=self.args.llm_tool_choice,
                # response_format={"type": "json_object"},
            )

            res = ""
            args_strs = []
            function_names = []
            is_tool_call = False
            tool_calls = []
            finish_reason = ""
            if self.args.llm_stream:
                for item in output:
                    if "finish_reason" in item['choices'][0]:
                        finish_reason = item['choices'][0]["finish_reason"]
                    if 'tool_calls' in item['choices'][0]['delta']:
                        is_tool_call = True
                        if item['choices'][0]['delta']['tool_calls'] is None or len(
                                item['choices'][0]['delta']['tool_calls']) == 0:
                            continue
                        tool_calls = item['choices'][0]['delta']['tool_calls']
                        for i in range(len(tool_calls)):
                            args_strs.append("")
                            function_names.append("")
                        for i, tool in enumerate(tool_calls):
                            args_strs[i] += tool["function"]["arguments"]
                            if tool["function"]['name'] is not None:
                                function_names[i] = tool["function"]['name']
                    else:
                        if 'content' in item['choices'][0]['delta']:
                            content = item['choices'][0]['delta']['content']
                            if content is None:
                                continue
                            res += content
                            pos = self._have_special_char(res)
                            if pos > -1:
                                yield res[:pos + 1]
                                res = res[pos + 1:]
                if len(res) > 0:
                    yield res
                if is_tool_call is True:
                    for i in range(len(tool_calls)):
                        tool_calls[i]["function"]["name"] = function_names[i]
                        tool_calls[i]["function"]["arguments"] = args_strs[i]
            else:
                if "finish_reason" in output['choices'][0]:
                    finish_reason = output['choices'][0]["finish_reason"]
                if 'tool_calls' in output['choices'][0]['message']:
                    is_tool_call = True
                    tool_calls = output['choices'][0]['message']['tool_calls']
                else:
                    res = output['choices'][0]['message']['content'] if 'content' in output['choices'][0]['message'] else ""
                    if res is not None:
                        yield res

            if is_tool_call is True:
                session.chat_history.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                })

                # @TODO: threading pool execute task
                for tool in tool_calls:
                    function_name = tool["function"]["name"]
                    args = json.loads(tool["function"]["arguments"])
                    func_res = FunctionManager.execute(function_name, session, **args)
                    logging.debug(f"tool calling: {function_name} {args} -> {func_res}")
                    # https://github.com/abetlen/llama-cpp-python/issues/1405
                    session.chat_history.append({
                        # "role": "tool",
                        "role": "function",
                        "tool_call_id": tool["id"],
                        "name": function_name,
                        "content": func_res,
                    })

            if finish_reason == "stop":
                break

    def _have_special_char(self, content: str) -> int:
        pattern = r"""[.。,，;；!！?？、]"""
        matches = re.findall(pattern, content)
        if len(matches) == 0:
            return -1
        return content.index(matches[len(matches) - 1])
