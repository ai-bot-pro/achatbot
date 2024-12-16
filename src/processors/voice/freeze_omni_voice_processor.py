import os
import logging
import sys
from typing import AsyncGenerator, Generator
import uuid

import torch
from apipeline.frames import *


from src.core.llm.transformers.manual_voice_freeze_omni import TransformersManualVoicFreezeOmni
from src.processors.voice.base import VoiceProcessorBase
from src.types.llm.lmgen import *
from src.types.llm.transformers import TransformersLMArgs
from src.common.session import Session
from src.common.types import SessionCtx
from src.common.utils.audio_utils import bytes2TorchTensorWith16
from src.types.frames import *


class FreezeOmniVoiceBaseProcessor(VoiceProcessorBase):
    """ """

    def __init__(
        self,
        *,
        system_prompt: str = "",
        voice_tokenizer_path: str | None = None,  # audio encoder/ft extractor
        model_path: str | None = None,  # gen lm and text tokenizer
        voice_decoder_path: str | None = None,  # audio decoder
        device: str = "cuda",
        torch_dtype: str = "auto",  # auto,float16,bfloat16,float32
        bnb_quant_type: str = "int4",
        session: Session | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        cur_dir = os.path.dirname(__file__)
        if bool(os.getenv("ACHATBOT_PKG", "")):
            sys.path.insert(1, os.path.join(cur_dir, "../../FreezeOmni"))
        else:
            sys.path.insert(1, os.path.join(cur_dir, "../../../deps/FreezeOmni"))

        self._sys_prompt = system_prompt or TransformersManualVoicFreezeOmni.DEFAULT_SYS_PROMPT
        self._voice_tokenizer_path = voice_tokenizer_path
        self._model_path = model_path
        self._voice_decoder_path = voice_decoder_path
        self._torch_dtype = torch_dtype
        self._bnb_quant_type = bnb_quant_type
        self._device = device

        self._session = session or Session(**SessionCtx(uuid.uuid4()).__dict__)

        self.reset()
        self.load_models()

    @property
    def stream_info(self) -> dict:
        """Return dict out stream info"""
        return {
            "sample_rate": self._voice_out_args.audio_sample_rate,
            "channels": self._voice_out_args.audio_channels,
        }

    def reset(self):
        # input_texts + completion_texts (audio tokenid special tag)
        self._history_texts = ""

    def load_models(self):
        logging.info("loading model weights")

        # TODO:

        logging.info("model weights loaded")

    async def start(self, frame: StartFrame):
        await super().start(frame)

        self._create_push_task()

        logging.info("start done")

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        logging.info("stop done")

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        logging.info("cancel done")

    async def run_voice(self, frame: AudioRawFrame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, PathAudioRawFrame):
            utt = frame.path
        else:
            audio_tensor = bytes2TorchTensorWith16(frame.audio)
            utt = (audio_tensor, self._voice_in_args.audio_sample_rate)

        # TODO: generate audio

        yield None
