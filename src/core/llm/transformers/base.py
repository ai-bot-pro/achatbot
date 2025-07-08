from abc import abstractmethod
import logging
from threading import Thread
import time

import torch

from src.common.session import Session
from src.common.interface import ILlm
from src.common.chat_history import ChatHistory
from src.core.llm.base import BaseLLM
from src.types.llm.transformers import TransformersLMArgs


class TransformersBaseLLM(BaseLLM, ILlm):
    def __init__(self, **args) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

        self.args = TransformersLMArgs(**args)

        if self.args.lm_torch_dtype != "auto":
            self.torch_dtype = getattr(torch, self.args.lm_torch_dtype)
        else:
            self.torch_dtype = "auto"

        if self.args.lm_device_map:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.args.lm_model_name_or_path,
                torch_dtype=self.args.lm_torch_dtype,
                #!NOTE: https://github.com/huggingface/transformers/issues/20896
                # device_map for multi cpu/gpu with accelerate
                device_map=self.args.lm_device_map,
                attn_implementation=self.args.lm_attn_impl,
                trust_remote_code=True,
            ).eval()
        else:
            self._model = (
                AutoModelForCausalLM.from_pretrained(
                    self.args.lm_model_name_or_path,
                    torch_dtype=self.args.lm_torch_dtype,
                    attn_implementation=self.args.lm_attn_impl,
                    trust_remote_code=True,
                )
                .eval()
                .to(self.args.lm_device)
            )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.args.lm_model_name_or_path, trust_remote_code=True
        )
        self._streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )

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

    @property
    def chat_history(self) -> ChatHistory:
        return self._chat_history if self._chat_history else ChatHistory()

    def set_system_prompt(self, **kwargs):
        pass

    def init(self):
        pass

    @abstractmethod
    def warmup(self):
        raise NotImplementedError("must be implemented in the child class")

    def generate(self, session: Session, **kwargs):
        r"""
        Instead of using model.chat(), we directly use model.generate()
        But you need to use tokenizer.apply_chat_template() to format your inputs as shown below
        !NOTE: session.ctx.state must have 'prompt' field with following format:
        for llm generate no chat template.
        - 'prompt': str (text+speech-tokens with instructions, no chat tpl)

        for llm chat template format.
        - 'prompt': str (text) # prompt or instruction
        - 'prompt': [PIL.Image,..., str] # vision, imgs+prompt
        - 'prompt': [str, np.ndarray] # voice, instruction+audio
        - 'prompt': [PIL.Image,..., np.ndarray] # vision+voice, instruction+audio+imgs
        - vision image 'prompt' e.g.: [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "desicribe this image"},
           ]
        - vision video 'prompt' e.g.: [
                {
                    "type": "video",
                    "video": "video_file.mp4",
                    "max_pixels": 360 * 420,
                    "fps": 1.0,
                },
                {"type": "text", "text": "Describe this video. Please reply to my message in chinese"},
            ]
        or
        - 'prompt': tuple (str, language_code)

        """
        pass

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
        return len(self._tokenizer.encode(text)) if self._tokenizer else 0

    def _warmup(self, target, args=(), kwargs=None, streamer=None):
        logging.info(f"Warming up {self.__class__.__name__} device: {self._model.device}")

        if "cuda" in str(self._model.device):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()

        n_steps = self.args.warmup_steps
        for step in range(n_steps):
            thread = Thread(target=target, args=args, kwargs=kwargs)
            thread.start()
            times = []
            start_time = time.perf_counter()
            for _ in streamer:
                times.append(time.perf_counter() - start_time)
                start_time = time.perf_counter()
            logging.info(f"step {step} warnup TTFT time: {times[0]} s")

        if "cuda" in str(self._model.device):
            end_event.record()
            torch.cuda.synchronize()
            logging.info(
                f"{self.__class__.__name__}:  warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s"
            )
