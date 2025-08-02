import os
import logging
import asyncio
from dataclasses import asdict
from typing import AsyncGenerator
from typing import Any, NamedTuple, Optional


try:
    from vllm import LLM, SamplingParams, EngineArgs
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error("you need to `pip install achatbot[vllm]`")
    raise Exception(f"Missing module: {e}")

from src.common.utils.audio_utils import bytes2NpArrayWith16
from src.common.session import Session
from src.modules.speech.asr.base import ASRBase


class WhisperVllmAsr(ASRBase):
    """
    - https://docs.vllm.ai/en/v0.7.0/getting_started/examples/whisper.html
    - https://docs.vllm.ai/en/stable/examples/offline_inference/audio_language.html
    """

    TAG = "whisper_vllm_asr"

    def __init__(self, **args) -> None:
        super().__init__(**args)

        engine_args = EngineArgs(
            model=self.args.model_name_or_path,
            max_model_len=448,
            max_num_seqs=5,
            limit_mm_per_prompt={
                "audio": 1,
                "image": 0,
                "video": 0,
            },
            # kv_cache_dtype="fp8", VLLM_USE_V1=1 is not supported with --kv-cache-dtype
        )

        # Create a Whisper encoder/decoder model instance
        # VLLM_USE_V1=1 is not supported with ['WhisperForConditionalGeneration']
        self.llm = LLM(**asdict(engine_args))

    async def transcribe_stream(self, session: Session) -> AsyncGenerator[str, None]:
        # stream word text @todo
        res = await self.transcribe(session)
        yield res["text"]

    async def transcribe(self, session: Session) -> dict:
        prompts = [
            {
                "prompt": self.args.prompt,
                "multi_modal_data": {"audio": (self.asr_audio.copy(), self.args.sample_rate)},
            },
        ]
        # https://docs.vllm.ai/en/stable/api/inference_params.html#vllm.SamplingParams
        sampling_params = SamplingParams(temperature=0.0, max_tokens=200, top_p=1.0)
        # Generate output tokens from the prompts. The output is a list of
        # RequestOutput objects that contain the prompt, generated
        # text, and other information.
        outputs = self.llm.generate(prompts, sampling_params)
        output = outputs[0]
        logging.debug(f"Output: {output}")
        generated_text = output.outputs[0].text

        res = {
            "language": self.args.language,
            "language_probability": None,
            "text": generated_text,
            "words": [],
        }
        return res
