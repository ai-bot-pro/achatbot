import logging
from threading import Thread


from common.session import Session
from src.common.interface import ILlm
from src.core.llm.base import BaseLLM
from src.types.llm.transformers import TransformersLLMArgs
from src.types.speech.language import TO_LLM_LANGUAGE


class TransformersManualLLM(BaseLLM, ILlm):
    TAG = "llm_transformers_manual"

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**TransformersLLMArgs().__dict__, **kwargs}

    def __init__(self, **args) -> None:
        from transformers import AutoModel, AutoTokenizer, TextIteratorStreamer

        self._args = TransformersLLMArgs(**args)
        self._model = AutoModel.from_pretrained(
            self._args.lm_model_name_or_path,
            torch_dtype=self._args.lm_torch_dtype,
            device_map=self._args.lm_device,
            attn_implementation=self._args.lm_attn_impl,
            trust_remote_code=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._args.lm_model_name_or_path, trust_remote_code=True)
        self._streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True)

    def generate(self, session: Session):
        r"""
        Instead of using model.chat(), we directly use model.generate()
        But you need to use tokenizer.apply_chat_template() to format your inputs as shown below
        !NOTE: session.ctx.state must have prompt field with following format:
        - 'prompt': str
        - 'prompt': [PIL.Image, str]
        - 'prompt': [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "描述下图片内容"},
           ]
        or
        - 'prompt': tuple (str, language_code)

        just for llm chat template format.
        """

        prompt = session.ctx.state['prompt']
        if isinstance(prompt, tuple):
            prompt, language_code = prompt
            if isinstance(prompt, str):
                prompt = f"Please reply to my message in {TO_LLM_LANGUAGE[language_code]}. " + prompt

        msgs = [{'role': 'user', 'content': prompt}]
        text = self._tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self._tokenizer(
            [text], return_tensors="pt").to(self._model.device)

        generation_kwargs = dict(
            model_inputs,
            streamer=self._streamer,
            max_new_tokens=self._args.max_new_tokens)
        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in self._streamer:
            yield new_text

    def chat_completion(self, session: Session):
        if self.args.llm_stream is False:
            res = ""
            for text in self._chat_stream(session):
                res += text
            yield res
        else:
            yield from self._chat_stream(session)

    def count_tokens(self, text: str | bytes):
        pass

    def _chat_stream(self, session: Session):
        prompt = session.ctx.state['prompt']
        if isinstance(prompt, tuple):
            prompt, language_code = prompt
            if isinstance(prompt, str):
                prompt = f"Please reply to my message in {TO_LLM_LANGUAGE[language_code]}. " + prompt

        msgs = [{'role': 'user', 'content': prompt}]
        # !NOTE: maybe some old version model don't support
        chat_kwargs = dict(
            image=None,
            msgs=msgs,
            tokenizer=self._tokenizer,
            streamer=self._streamer,
            max_new_tokens=self._args.max_new_tokens)
        thread = Thread(target=self._model.chat, kwargs=chat_kwargs)
        thread.start()

        for new_text in self._streamer:
            yield new_text
