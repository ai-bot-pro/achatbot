import logging
from typing import AsyncGenerator

import torch
import numpy as np

from src.common.types import PYAUDIO_PAINT16
from src.common.interface import ITts
from src.common.session import Session
from src.types.speech.tts.chat import ChatTTSArgs
from src.common.utils import audio_utils
from .base import BaseTTS


class ChatTTS(BaseTTS, ITts):
    TAG = "tts_chat"

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**ChatTTSArgs().__dict__, **kwargs}

    def __init__(self, **args) -> None:
        import deps.ChatTTS.ChatTTS as ChatTTS

        self.args = ChatTTSArgs(**args)
        self.chat = ChatTTS.Chat()
        self.chat.load(
            source=self.args.source,
            force_redownload=self.args.force_redownload,
            custom_path=self.args.local_path,
            compile=self.args.compile,
            device=self.args.device,
            use_flash_attn=self.args.use_flash_attn,
        )

        self.rand_speaker = self.chat.sample_random_speaker()
        self.args.params_infer_code = ChatTTS.Chat.InferCodeParams(
            # Sample a speaker from Gaussian.
            spk_emb=None,
            temperature=0.3,
            top_P=0.7,
            top_K=20,
            repetition_penalty=1.05,
            max_new_token=2048,
        )

        # For sentence level manual control.
        # use oral_(0-9), laugh_(0-2), break_(0-7)
        # to generate special token in text to synthesize.
        self.args.params_refine_text = ChatTTS.Chat.RefineTextParams(
            prompt="[oral_2][laugh_0][break_6]",
            max_new_token=1024,
        )

    def set_voice(self, speaker: str) -> None:
        self.args.params_infer_code.spk_emb = speaker

    def get_stream_info(self) -> dict:
        return {
            "format": PYAUDIO_PAINT16,
            "channels": 1,
            "rate": 24000,
            "sample_width": 2,
            "np_dtype": np.int16,
        }

    async def _inference(
        self, session: Session, text: str, **kwargs
    ) -> AsyncGenerator[bytes, None]:
        self.set_voice(self.rand_speaker)
        logging.debug(f"{self.TAG} synthesis: {text}")
        wav = self.chat.infer(
            [
                text,
            ],
            skip_refine_text=self.args.skip_refine_text,
            refine_text_only=self.args.refine_text_only,
            params_refine_text=self.args.params_refine_text,
            params_infer_code=self.args.params_infer_code,
            use_decoder=self.args.use_decoder,
            do_text_normalization=self.args.do_text_normalization,
            lang=self.args.lang,
            stream=self.args.tts_stream,
        )

        if self.args.tts_stream is False:
            yield audio_utils.postprocess_tts_wave_int16(torch.from_numpy(wav[0]))
        else:
            for chunk in wav:
                yield audio_utils.postprocess_tts_wave_int16(torch.from_numpy(chunk))
