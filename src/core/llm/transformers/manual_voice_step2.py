import time
import logging

import torch

from src.common.session import Session
from src.types.llm.transformers import TransformersLMArgs
from .base import TransformersBaseLLM
from .models.step_audio2 import StepAudio2Stream


class TransformersManualVoiceStep2(TransformersBaseLLM):
    """
    https://huggingface.co/stepfun-ai/Step-Audio-2-mini
    """

    TAG = "llm_transformers_manual_voice_step2"
    RATE = 24000

    def __init__(self, **args):
        self.args = TransformersLMArgs(**args)
        self._audio_llm = StepAudio2Stream(model_path=self.args.lm_model_name_or_path)
        self.eos_token_id = [
            self._audio_llm.eos_token_id,
            self._audio_llm.llm_tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        ]
        self.warmup()

    @property
    def llm(self):
        return self._audio_llm

    @property
    def llm_tokenizer(self):
        return self._audio_llm.llm_tokenizer

    @torch.inference_mode()
    def warmup(self):
        if self.args.warmup_steps < 1:
            return

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "human", "content": self.args.warmup_prompt},
            {"role": "assistant", "content": None},
        ]
        for step in range(self.args.warmup_steps):
            token_iter = self._audio_llm(
                messages,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                eos_token_id=self.eos_token_id,
            )
            first = True
            start = time.time()
            for _ in token_iter:
                if first:
                    first = False
                    ttft = time.time() - start
            total_time = time.time() - start
            logging.info(f"warmup {step=} {ttft=:.3f}s {total_time=:.3f}s")

    @torch.inference_mode()
    def generate(self, session: Session, **kwargs):
        for token_id in self._audio_llm(
            messages=session.ctx.state["messages"],
            **kwargs,
        ):
            yield token_id
