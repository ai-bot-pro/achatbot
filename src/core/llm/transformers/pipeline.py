import logging
from threading import Thread

import torch

from common.session import Session
from src.common.interface import ILlm
from src.core.llm.base import BaseLLM
from src.types.llm.transformers import TransformersLLMArgs
from src.types.speech.language import TO_LLM_LANGUAGE


class TransformersPipelineLLM(BaseLLM, ILlm):
    TAG = "llm_transformers_pipeline"

    def __init__(self, **args) -> None:
        from transformers import AutoModel, AutoTokenizer, pipeline, TextIteratorStreamer

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
        self._pipe = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            device=self.args.lm_device)
        self._streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True)

        self.warmup()

    def warmup(self):
        logging.info(f"Warming up {self.__class__.__name__}")

        dummy_input_text = self.args.warnup_prompt
        dummy_msgs = [{"role": self.args.user_role, "content": dummy_input_text}]

        warmup_gen_kwargs = dict(
            dummy_msgs,
            streamer=self._streamer,
            return_full_text=False,
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
            thread = Thread(target=self._pipe, kwargs=warmup_gen_kwargs)
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

        generation_kwargs = dict(
            msgs,
            return_full_text=False,
            streamer=self._streamer,
            min_new_tokens=self.args.lm_gen_min_new_tokens,
            max_new_tokens=self.args.lm_gen_max_new_tokens)
        thread = Thread(target=self._pipe, kwargs=generation_kwargs)
        thread.start()

        for new_text in self._streamer:
            yield new_text

    def chat_completion(self, session: Session):
        if self.args.llm_stream is False:
            res = ""
            for text in self.generate(session):
                res += text
            yield res
        else:
            yield from self.generate(session)

    def count_tokens(self, text: str | bytes):
        pass
