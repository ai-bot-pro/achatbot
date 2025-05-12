import logging
from typing import AsyncGenerator

import numpy as np

from src.core.llm.transformers.manual_voice_vita import (
    TransformersManualTextSpeechVITALLM,
)
from src.common.types import PYAUDIO_PAFLOAT32, PYAUDIO_PAINT16
from src.common.interface import ITts
from src.common.session import Session
from .base import BaseTTS


class VITATTS(BaseTTS, ITts):
    """ """

    TAG = "tts_vita"

    def __init__(self, **kwargs) -> None:
        kwargs["audio_tokenizer_model_path"] = None
        kwargs["sense_voice_model_path"] = None
        self.lm_model = TransformersManualTextSpeechVITALLM(**kwargs)

    def get_stream_info(self) -> dict:
        return {
            # "format": PYAUDIO_PAINT16,
            "format": PYAUDIO_PAFLOAT32,
            "channels": 1,
            "rate": TransformersManualTextSpeechVITALLM.RATE,
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
        session.ctx.state["message"] = "Convert the text to speech.\n" + input_text
        kwargs["do_sample"] = True
        kwargs["mode"] = None
        tensor_audio_stream = self.lm_model.generate(session, **kwargs)

        for tensor_audio_dict in tensor_audio_stream:
            if (
                tensor_audio_dict is not None and "audio_wav" in tensor_audio_dict
            ):  # don't use if tensor_audio to check
                audio_tensor = tensor_audio_dict["audio_wav"]
                audio_np = audio_tensor.squeeze(0).float().detach().cpu().numpy()
                # audio_np = (audio_np * 32767).astype(np.int16)
                yield audio_np.tobytes()
        yield None  # end of stream
