import time
import logging
from dataclasses import dataclass, field

import numpy as np
from typing import AsyncGenerator, Iterator

from src.common.session import Session
from src.common.interface import ILlm
from src.common.chat_history import ChatHistory
from src.core.llm.base import BaseLLM
from src.common.logger import Logger
from src.common.utils import task
from src.types.llm.sampling import LMGenerateArgs
from .client.step_audio2_mini_vllm import StepAudio2MiniVLLMClient
from . import VLlmBase


@dataclass
class VllmStepAudio2Args:
    api_url: str = field(
        default="http://127.0.0.1:8000/v1/chat/completions",
        metadata={"help": "vllm api url. Default is 'http://127.0.0.1:8000/v1/chat/completions'."},
    )
    model_name: str = field(
        default="step-audio-2-mini",
        metadata={"help": "vllm register model name. Default is 'step-audio-2-mini'."},
    )
    lm_model_name_or_path: str = field(
        default=None,
        metadata={"help": "The pretrained language model to use. Default is None."},
    )
    warmup_prompt: str = field(
        default="Repeat the word 'weedge niu bi'.",
        metadata={"help": "warmup llm generate prompt. Default is 'weedge niu bi'."},
    )
    warmup_steps: int = field(
        default=2,
        metadata={"help": "The number of steps to run the warmup prompt. Default is 2."},
    )
    verbose: bool = field(
        default=False,
        metadata={"help": "Whether to print debug info. Default is False."},
    )
    gen_args: dict = field(default_factory=lambda: LMGenerateArgs().__dict__)

    def update(self, **kwargs):
        unused_kwargs = dict()
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                unused_kwargs[key] = value
        return unused_kwargs


class VllmClientStepAudio2(VLlmBase):
    TAG = "llm_vllm_client_step_audio2"
    RATE = 24000

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.args = VllmStepAudio2Args()
        self.args.update(**kwargs)
        self.gen_args = LMGenerateArgs()
        self.gen_args.update(**self.args.gen_args)
        logging.info(f"{self.args=} {self.gen_args=}")

        self.client = StepAudio2MiniVLLMClient(
            self.args.api_url,
            self.args.model_name,
            tokenizer_path=self.args.lm_model_name_or_path,
        )

        self.warmup()

    @property
    def llm_tokenizer(self):
        return self.client.llm_tokenizer

    def warmup(self):
        if self.args.warmup_steps <= 0:
            return
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "human", "content": self.args.warmup_prompt},
            {"role": "assistant", "content": None},
        ]
        for step in range(self.args.warmup_steps):
            stream_iter = self.client.stream(
                messages,
                stream=True,
                max_tokens=128,
                temperature=0.7,
                top_p=0.9,
                frequency_penalty=0,
                repetition_penalty=1.05,
                skip_special_tokens=False,
                parallel_tool_calls=False,
            )
            first = True
            start = time.time()
            for _ in stream_iter:
                if first:
                    first = False
                    ttft = time.time() - start
            total_time = time.time() - start
            logging.info(f"warmup {step=} {ttft=:.3f}s {total_time=:.3f}s")

    def generate(self, session: Session, **kwargs) -> Iterator[str | dict | np.ndarray]:
        messages = session.ctx.state.get("messages", [])
        stop_ids = kwargs.pop("stop_ids", self.gen_args.lm_gen_stop_ids)
        max_completion_tokens = kwargs.get("max_completion_tokens") or kwargs.pop(
            "max_new_tokens", self.gen_args.lm_gen_max_new_tokens
        )
        temperature = kwargs.pop("temperature", self.gen_args.lm_gen_temperature)
        top_p = kwargs.pop("top_p", self.gen_args.lm_gen_top_p)
        top_k = kwargs.pop("top_k", self.gen_args.lm_gen_top_k)
        min_p = kwargs.pop("min_p", self.gen_args.lm_gen_min_p)
        repetition_penalty = kwargs.pop(
            "repetition_penalty", self.gen_args.lm_gen_repetition_penalty
        )
        stream_iter = self.client.stream(
            messages,
            stream=True,
            max_completion_tokens=max_completion_tokens,
            skip_special_tokens=False,
            parallel_tool_calls=False,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )
        stop = False

        tool_calls = []
        for response, text, audio, token_ids in stream_iter:
            if self.args.verbose is True:
                print(f"{response=} {text=} {audio=} {token_ids=}")
            if len(response.get("tool_calls", [])) > 0:
                if len(tool_calls) == 0:
                    tool_calls = response.get("tool_calls")
                else:
                    tool_calls[0]["function"]["arguments"] = response["tool_calls"][0]["function"][
                        "arguments"
                    ]
            if token_ids is None:
                continue
            for token_id in token_ids:
                if token_id in stop_ids:
                    stop = True
                    break
                yield token_id
            if stop:
                break
        yield {"tool_calls": tool_calls}
