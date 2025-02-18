import logging
from typing import AsyncGenerator

import numpy as np

from src.core.llm.transformers.manual_vision_voice_minicpmo import (
    TransformersManualInstructSpeechMiniCPMO,
)
from src.common.types import PYAUDIO_PAFLOAT32
from src.common.interface import ITts
from src.common.session import Session
from .base import BaseTTS


class MiniCPMoTTS(BaseTTS, ITts):
    """
    tts the same as chattts (Chattts-200M), but add use qwen2.5 as lm to gen text with tts task instruction
    """

    TAG = "tts_minicpmo"
    DEFAULT_INSTRUCT_TPL = f"Speak like a male charming superstar, radiating confidence and style in every word. Please read the text below:\n"
    DEFAULT_VOICE_CLONE_INSTRUCT = f"Please read the text below."

    def __init__(self, **kwargs) -> None:
        self.instruct_tpl = self.DEFAULT_INSTRUCT_TPL
        instruct_tpl = kwargs.pop("instruct_tpl", self.DEFAULT_INSTRUCT_TPL)
        if len(instruct_tpl.strip()) > 0:
            self.instruct_tpl = instruct_tpl
        else:
            self.instruct_tpl = self.DEFAULT_INSTRUCT_TPL
        voice_clone_instruct = kwargs.pop("voice_clone_instruct", self.DEFAULT_VOICE_CLONE_INSTRUCT)
        if len(voice_clone_instruct.strip()) > 0:
            self.voice_clone_instruct = voice_clone_instruct
        else:
            self.voice_clone_instruct = self.DEFAULT_VOICE_CLONE_INSTRUCT
        self.lm_model = TransformersManualInstructSpeechMiniCPMO(**kwargs)

    def get_stream_info(self) -> dict:
        return {
            # "format": PYAUDIO_PAINT16,
            "format": PYAUDIO_PAFLOAT32,
            "channels": 1,
            "rate": TransformersManualInstructSpeechMiniCPMO.RATE,
            "sample_width": 2,
            # "np_dtype": np.int16,
            "np_dtype": np.float32,
        }

    async def _inference(
        self, session: Session, text: str, **kwargs
    ) -> AsyncGenerator[bytes, None]:
        input_text = text.strip()
        if len(input_text) == 0:
            yield None
            return
        is_instruct = kwargs.pop("is_instruct", False)
        instruction = self.instruct_tpl + input_text
        if is_instruct is True:
            instruction = input_text
        session.ctx.state["prompt"] = [instruction]
        if self.lm_model.tts_task == "voice_cloning":
            voice_clone_instruct = kwargs.pop("voice_clone_instruct", self.voice_clone_instruct)
            session.ctx.state["prompt"] = [voice_clone_instruct, input_text]
        tensor_audio_stream = self.lm_model.generate(session, **kwargs)

        for tensor_audio in tensor_audio_stream:
            if tensor_audio is not None:  # don't use if tensor_audio to check
                yield tensor_audio.float().detach().cpu().numpy().tobytes()
            else:
                yield None
