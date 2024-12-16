import os
import logging
import sys
from typing import AsyncGenerator, Generator
import uuid

import torch
from apipeline.frames import *


from src.processors.voice.base import VoiceProcessorBase
from src.types.llm.lmgen import *
from src.types.llm.transformers import TransformersLMArgs
from src.common.session import Session
from src.common.types import SessionCtx
from src.common.utils.audio_utils import bytes2TorchTensorWith16
from src.types.frames import *

DEFAULT_SYS_PROMPT = "You are a helpful voice assistant.\
Your answer should be coherent, natural, simple, complete.\
Your name is Xiao Yun.\
Your inventor is Tencent."


class FreezeOmniVoiceBaseProcessor(VoiceProcessorBase):
    """ """

    def __init__(
        self,
        *,
        system_prompt: str = "",
        model_path: str | None = None,  # audio-llm, decoder and codec(decoder) ckpt path
        llm_path: str | None = None,  # text llm path
        top_k: int = 20,
        top_p: float = 0.8,
        temperature: float = 0.8,
        session: Session | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        cur_dir = os.path.dirname(__file__)
        if bool(os.getenv("ACHATBOT_PKG", "")):
            sys.path.insert(1, os.path.join(cur_dir, "../../FreezeOmni"))
        else:
            sys.path.insert(1, os.path.join(cur_dir, "../../../deps/FreezeOmni"))

        self._sys_prompt = system_prompt or DEFAULT_SYS_PROMPT
        self._model_path = model_path
        self._llm_path = llm_path
        self._top_k = top_k
        self._top_p = top_p
        self._temperature = temperature

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
