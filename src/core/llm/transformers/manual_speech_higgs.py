import os
import sys
import logging
import time
from typing import AsyncGenerator

import numpy as np

try:
    cur_dir = os.path.dirname(__file__)
    if bool(os.getenv("ACHATBOT_PKG", "")):
        sys.path.insert(0, os.path.join(cur_dir, "../../../HiggsAudio"))
    else:
        sys.path.insert(0, os.path.join(cur_dir, "../../../../deps/HiggsAudio"))
    from deps.HiggsAudio.boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
    from deps.HiggsAudio.boson_multimodal.data_types import ChatMLSample, Message, AudioContent
    from deps.HiggsAudio.boson_multimodal.model.higgs_audio.utils import revert_delay_pattern

except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use Higgs-Audio, you need to `pip install achatbot[llm_transformers_manual_speech_higgs]`,"
    )
    raise Exception(f"Missing module: {e}")

from src.common.utils.helper import get_device
from src.common.session import Session
from src.types.llm.transformers import TransformersSpeechLMArgs
from .base import TransformersBaseLLM


class TransformersManualSpeechHiggs(TransformersBaseLLM):
    """
    TTS: text + ref audio -> llama3 -> text + vq code tokens
    """

    TAG = "llm_transformers_manual_speech_higgs"
    DEFAULT_SYS_PROMPT = "Generate audio following instruction.\n\n<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>"

    def __init__(self, **args):
        audio_tokenizer_path = args.pop("audio_tokenizer_path", None)
        self.args = TransformersSpeechLMArgs(**args)
        self.args.lm_device = self.args.lm_device or get_device()
        logging.info("TransformersLMArgs: %s", self.args)

        self.serve_engine = HiggsAudioServeEngine(
            self.args.lm_model_name_or_path,
            audio_tokenizer_path,
            device=self.args.lm_device,
            torch_dtype=self.args.lm_torch_dtype,
        )

        self.warmup()

    def warmup(self):
        if self.args.warmup_steps <= 0 or not self.args.warmup_prompt:
            logging.info("no warmup!")
            return

        messages = [
            Message(
                role="system",
                content=self.DEFAULT_SYS_PROMPT,
            ),
            Message(
                role="user",
                content=self.args.warmup_prompt,
            ),
        ]
        for step in range(self.args.warmup_steps):
            start_time = time.perf_counter()
            _ = self.serve_engine.generate(
                chat_ml_sample=ChatMLSample(messages=messages),
                max_new_tokens=self.args.lm_gen_max_new_tokens,
                temperature=self.args.lm_gen_temperature,
                top_p=self.args.lm_gen_top_p,
                top_k=self.args.lm_gen_top_k,
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
            )
            cost = time.perf_counter() - start_time
            logging.info(f"step {step} | warmup cost: {cost:.3f} s")

    async def async_generate(
        self, session: Session, **kwargs
    ) -> AsyncGenerator[str | dict | np.ndarray, None]:
        messages = session.ctx.state.get("messages", [])
        assert isinstance(messages, list) and len(messages) > 0, (
            "messages should be a list of Message objects"
        )
        streamer = self.serve_engine.generate_delta_stream(
            chat_ml_sample=ChatMLSample(messages=messages),
            max_new_tokens=kwargs.get("max_new_tokens", self.args.lm_gen_max_new_tokens),
            temperature=kwargs.get("temperature", self.args.lm_gen_temperature),
            top_p=kwargs.get("top_p", self.args.lm_gen_top_p),
            top_k=kwargs.get("top_k", self.args.lm_gen_top_k),
            stop_strings=kwargs.get("stop_strings", ["<|end_of_text|>", "<|eot_id|>"]),
        )

        times = []
        start_time = time.perf_counter()
        async for delta in streamer:
            times.append(time.perf_counter() - start_time)
            yield {
                "text": delta.text,
                "text_tokens": delta.text_tokens,
                "audio_vq_tokens": delta.audio_tokens,
                "finish_reason": delta.finish_reason,
            }
            start_time = time.perf_counter()

        if len(times) > 0:
            logging.info(f"gen TTFT time: {times[0]} s | total: {sum(times)} s")
        else:
            logging.warning("no generate stream")
