import logging
from threading import Thread

import torch

from common.session import Session
from src.common.interface import ILlm
from src.core.llm.base import BaseLLM
from src.types.llm.transformers import TransformersLLMArgs
from src.types.speech.language import TO_LLM_LANGUAGE


class TransformersManualLLM(BaseLLM, ILlm):
    TAG = "llm_transformers_manual"

    def __init__(self, **args) -> None:
        from transformers import AutoModel, AutoTokenizer, TextIteratorStreamer

        if self.args.lm_torch_dtype != "auto":
            self.torch_dtype = getattr(torch, self.args.lm_torch_dtype)
        else:
            self.torch_dtype = "auto"

        self.args = TransformersLLMArgs(**args)
        if self.args.lm_device_map:
            self._model = AutoModel.from_pretrained(
                self.args.lm_model_name_or_path,
                torch_dtype=self.args.lm_torch_dtype,
                #!NOTE: https://github.com/huggingface/transformers/issues/20896
                # device_map for multi cpu/gpu with accelerate
                device_map=self.args.lm_device_map,
                attn_implementation=self.args.lm_attn_impl,
                trust_remote_code=True,
            )
        else:
            self._model = AutoModel.from_pretrained(
                self.args.lm_model_name_or_path,
                torch_dtype=self.args.lm_torch_dtype,
                attn_implementation=self.args.lm_attn_impl,
                trust_remote_code=True,
            ).eval().to(self.args.lm_device)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.args.lm_model_name_or_path, trust_remote_code=True)
        self._streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True)

        self.warmup()

    def warmup(self):
        logging.info(f"Warming up {self.__class__.__name__}")

        dummy_input_text = self.args.warnup_prompt
        dummy_msgs = [{"role": self.args.user_role, "content": dummy_input_text}]
        text = self._tokenizer.apply_chat_template(
            dummy_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self._tokenizer(
            [text], return_tensors="pt").to(self._model.device)

        warmup_gen_kwargs = dict(
            model_inputs,
            streamer=self._streamer,
            min_new_tokens=self.args.lm_gen_min_new_tokens,
            max_new_tokens=self.args.lm_gen_max_new_tokens,
        )

        if self.args.lm_device_map is None and self.args.lm_device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()

        n_steps = self.args.warnup_steps
        for _ in range(n_steps):
            thread = Thread(target=self._mode.generate, kwargs=warmup_gen_kwargs)
            thread.start()
            for _ in self.streamer:
                pass

        if self.args.lm_device_map is None and self.args.lm_device == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            logging.info(
                f"{self.__class__.__name__}:  warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s"
            )

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

        msgs = [{'role': self.args.user_role, 'content': prompt}]
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
            min_new_tokens=self.args.lm_gen_min_new_tokens,
            max_new_tokens=self.args.lm_gen_max_new_tokens)
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

        msgs = [{'role': self.args.user_role, 'content': prompt}]
        # !NOTE: maybe some old version model don't support
        chat_kwargs = dict(
            image=None,
            msgs=msgs,
            tokenizer=self._tokenizer,
            streamer=self._streamer,
            min_new_tokens=self.args.min_new_tokens,
            max_new_tokens=self.args.max_new_tokens)
        thread = Thread(target=self._model.chat, kwargs=chat_kwargs)
        thread.start()

        for new_text in self._streamer:
            yield new_text
