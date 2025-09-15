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
from src.common.types import SessionCtx
from src.types.llm.sampling import LMGenerateArgs
from .client.step_audio2_mini_vllm import StepAudio2MiniVLLMClient
from .base import VLlmBase


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
    tokenizer_path: str = field(
        default=None,
        metadata={"help": "vllm llm tokenizer path for text tokenization. Default is None."},
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


class VllmClientStepAudio2(VLlmBase):
    TAG = "llm_vllm_client_step_audio2"
    RATE = 24000

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.args = VllmStepAudio2Args(**kwargs)
        self.gen_args = self.args.gen_args

        self.client = StepAudio2MiniVLLMClient(
            self.args.api_url,
            self.args.model_name,
            tokenizer_path=self.args.tokenizer_path,
        )

    @property
    def llm_tokenizer(self):
        return self.client.llm_tokenizer

    async def warmup(self):
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
        stop_ids = kwargs.pop("stop_ids", self.args.lm_gen_stop_ids)
        stream_iter = self.client.stream(
            messages,
            stream=True,
            **kwargs,
        )
        stop = False
        for response, text, audio, token_ids in stream_iter:
            if self.args.verbose is True:
                print(f"{response=} {text=} {audio=} {token_ids=}")
            if token_ids is None:
                continue
            for token_id in token_ids:
                if token_id in stop_ids:
                    stop = True
                    break
                yield token_id
            if stop:
                break
