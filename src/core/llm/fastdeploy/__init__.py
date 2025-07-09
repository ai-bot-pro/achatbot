from io import BytesIO
import logging
import json
import re
import os


from src.common.interface import ILlm
from src.common.session import Session
from src.common.types import DEFAULT_SYSTEM_PROMPT, LLamcppLLMArgs
from src.modules.functions.function import FunctionManager
from ..base import BaseLLM


class PromptInit:
    """
    Note: for generate model
    """

    @staticmethod
    def create_phi3_prompt(history: list[str], system_prompt: str, init_message: str = None):
        prompt = f"<|system|>\n{system_prompt}</s>\n"
        if init_message:
            prompt += f"<|assistant|>\n{init_message}</s>\n"

        return prompt + "".join(history) + "<|assistant|>\n"

    def create_qwen_prompt(history: list[str], system_prompt: str, init_message: str = None):
        prompt = f"<|system|>\n{system_prompt}<|end|>\n"
        if init_message:
            prompt += f"<|assistant|>\n{init_message}<|end|>\n"

        return prompt + "".join(history) + "<|assistant|>\n"

    @staticmethod
    def create_prompt(name: str, history: list[str], init_message: str = None):
        system_prompt = os.getenv("LLM_CHAT_SYSTEM", DEFAULT_SYSTEM_PROMPT)
        if "phi-3" == name:
            return PromptInit.create_phi3_prompt(history, system_prompt, init_message)
        if "qwen-2" == name:
            return PromptInit.create_qwen_prompt(history, system_prompt, init_message)

        return None

    @staticmethod
    def get_user_prompt(name: str, text: str):
        if "phi-3" == name:
            return f"<|user|>\n{text}</s>\n"
        if "qwen-2" == name:
            return f"<|start|>user\n{text}<|end|>\n"
        return None

    @staticmethod
    def get_assistant_prompt(name: str, text: str):
        if "phi-3" == name:
            return f"<|assistant|>\n{text}</s>\n"
        if "qwen-2" == name:
            return f"<|assistant|>\n{text}<|end|>\n"
        return None


class LLamacppLLM(BaseLLM, ILlm):
    TAG = ["llm_llamacpp", "llm_llamacpp_vision"]

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**LLamcppLLMArgs().__dict__, **kwargs}

    def get_chat_handler(self):
        match self.args.chat_format:
            case "minicpm-v-2.6":
                from llama_cpp.llama_chat_format import MiniCPMv26ChatHandler

                return MiniCPMv26ChatHandler(clip_model_path=self.args.clip_model_path)
            case "llava-1-5":
                from llama_cpp.llama_chat_format import Llava15ChatHandler

                return Llava15ChatHandler(clip_model_path=self.args.clip_model_path)
            case "llava-1-6":
                from llama_cpp.llama_chat_format import Llava16ChatHandler

                return Llava16ChatHandler(clip_model_path=self.args.clip_model_path)
            case "moondream2":
                from llama_cpp.llama_chat_format import MoondreamChatHandler

                return MoondreamChatHandler(clip_model_path=self.args.clip_model_path)
            case "nanollava":
                from llama_cpp.llama_chat_format import NanoLlavaChatHandler

                return NanoLlavaChatHandler(clip_model_path=self.args.clip_model_path)
            case "llama-3-vision-alpha":
                from llama_cpp.llama_chat_format import Llama3VisionAlphaChatHandler

                return Llama3VisionAlphaChatHandler(clip_model_path=self.args.clip_model_path)
            case _:
                return None

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
                chat_handler=self.get_chat_handler(),
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
                chat_handler=self.get_chat_handler(),
            )

        self.warmup()

    def warmup(self):
        pass

    def encode(self, text: str | bytes):
        return self.model.tokenize(text.encode() if isinstance(text, str) else text)

    def count_tokens(self, text: str | bytes):
        return len(self.encode(text))

    def generate(self, session: Session):
        prompt = session.ctx.state["prompt"]
        if isinstance(prompt, str) and self.args.llm_prompt_tpl:
            prompt = self.args.llm_prompt_tpl % (prompt,)
        # https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_completion
        output = self.model.create_completion(
            prompt,
            max_tokens=self.args.llm_max_tokens,  # Generate up to 256 tokens
            stop=self.args.llm_stop,
            # echo=True,  # Whether to echo the prompt
            stream=self.args.llm_stream,
            temperature=self.args.llm_temperature,
            top_p=self.args.llm_top_p,
            top_k=self.args.llm_top_k,
            seed=self.args.llm_seed,
        )
        logging.debug(f"llm generate: {output}")
        res = ""
        if self.args.llm_stream:
            for item in output:
                content = item["choices"][0]["text"]
                res += content
                pos = self._have_special_char(res)
                if pos > -1:
                    yield res[: pos + 1]
                    res = res[pos + 1 :]
            if len(res) > 0:
                yield res
        else:
            yield output["choices"][0]["text"]

    def chat_completion(self, session: Session):
        if self.args.model_type not in ["chat", "chat-func"]:
            yield from self.generate(session)
            return
        query = session.ctx.state["prompt"]
        self.args.save_chat_history and session.chat_history.append(
            {"role": "user", "content": query}
        )
        res = ""
        if self.args.model_type == "chat":
            for item in self._chat_completion(session):
                res += item
                yield item
        elif self.args.model_type == "chat-func":
            for item in self._chat_completion_functions(session):
                res += item
                yield item
        self.args.save_chat_history and session.chat_history.append(
            {"role": "assistant", "content": res}
        )
        logging.debug(f"chat_history:{session.chat_history}")

    def _chat_completion(self, session: Session):
        logging.debug(f"chat_history:{session.chat_history}")
        # https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_chat_completion
        output = self.model.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": self.args.llm_chat_system,
                },
                *session.chat_history,
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
                if "content" in item["choices"][0]["delta"]:
                    content = item["choices"][0]["delta"]["content"]
                    if content is None:
                        continue
                    res += content
                    pos = self._have_special_char(res)
                    if pos > -1:
                        yield res[: pos + 1]
                        res = res[pos + 1 :]
            if len(res) > 0:
                yield res
        else:
            res = (
                output["choices"][0]["message"]["content"]
                if "content" in output["choices"][0]["message"]
                else ""
            )
            if res is not None:
                yield res

    def _chat_completion_functions(self, session: Session):
        tools = FunctionManager.get_tool_calls()
        logging.info(f"tools: {tools}")
        while True:
            messages = [
                {
                    "role": "system",
                    "content": self.args.llm_chat_system,
                },
                *session.chat_history,
            ]
            logging.info(f"messages: {messages}")
            # https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_chat_completion
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
                    if "finish_reason" in item["choices"][0]:
                        finish_reason = item["choices"][0]["finish_reason"]
                    if "tool_calls" in item["choices"][0]["delta"]:
                        is_tool_call = True
                        if (
                            item["choices"][0]["delta"]["tool_calls"] is None
                            or len(item["choices"][0]["delta"]["tool_calls"]) == 0
                        ):
                            continue
                        tool_calls = item["choices"][0]["delta"]["tool_calls"]
                        for i in range(len(tool_calls)):
                            args_strs.append("")
                            function_names.append("")
                        for i, tool in enumerate(tool_calls):
                            args_strs[i] += tool["function"]["arguments"]
                            if tool["function"]["name"] is not None:
                                function_names[i] = tool["function"]["name"]
                    else:
                        if "content" in item["choices"][0]["delta"]:
                            content = item["choices"][0]["delta"]["content"]
                            if content is None:
                                continue
                            res += content
                            pos = self._have_special_char(res)
                            if pos > -1:
                                yield res[: pos + 1]
                                res = res[pos + 1 :]
                if len(res) > 0:
                    yield res
                if is_tool_call is True:
                    for i in range(len(tool_calls)):
                        tool_calls[i]["function"]["name"] = function_names[i]
                        tool_calls[i]["function"]["arguments"] = args_strs[i]
            else:
                if "finish_reason" in output["choices"][0]:
                    finish_reason = output["choices"][0]["finish_reason"]
                if "tool_calls" in output["choices"][0]["message"]:
                    is_tool_call = True
                    tool_calls = output["choices"][0]["message"]["tool_calls"]
                else:
                    res = (
                        output["choices"][0]["message"]["content"]
                        if "content" in output["choices"][0]["message"]
                        else ""
                    )
                    if res is not None:
                        yield res

            if is_tool_call is True:
                self.args.save_chat_history and session.chat_history.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls,
                    }
                )

                # @TODO: threading pool execute task
                for tool in tool_calls:
                    function_name = tool["function"]["name"]
                    args = json.loads(tool["function"]["arguments"])
                    func_res = FunctionManager.execute(function_name, session, **args)
                    logging.debug(f"tool calling: {function_name} {args} -> {func_res}")
                    # https://github.com/abetlen/llama-cpp-python/issues/1405
                    self.args.save_chat_history and session.chat_history.append(
                        {
                            # "role": "tool",
                            "role": "function",
                            "tool_call_id": tool["id"],
                            "name": function_name,
                            "content": func_res,
                        }
                    )

            if finish_reason == "stop":
                break
