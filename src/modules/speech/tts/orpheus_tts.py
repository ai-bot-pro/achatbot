from dataclasses import dataclass, field
import logging
import math
import random
import os
import re
import sys
from typing import AsyncGenerator, List

from dotenv import load_dotenv
import numpy as np
import torch

from src.core.llm.transformers.manual_speech_llama import TransformersManualSpeechLlama
from src.common.random import set_all_random_seed
from src.common.interface import ITts
from src.common.session import Session
from src.common.types import PYAUDIO_PAFLOAT32
from src.types.speech.tts.orpheus import OrpheusTTSArgs
from src.types.llm.transformers import TransformersSpeechLMArgs
from src.types.codec import CodecArgs
from .base import BaseTTS

load_dotenv(override=True)

try:
    from src.modules.codec.audio.snac import SNACCodec
except ModuleNotFoundError as e:
    logging.error(
        "In order to use orpheus tts, you need to `pip install achatbot[tts_orpheus]`.\nPlease install the missing modules."
    )
    raise Exception(
        f"Missing module: {e}. Please run `pip install achatbot[tts_orpheus]` to install the dependencies."
    )


@dataclass
class RefAudioCodecInfo:
    ref_speaker: str = ""
    ref_text: str = ""
    ref_path: str = ""
    vq_indices: List[torch.Tensor] = field(default_factory=list)


class OrpheusTTS(BaseTTS, ITts):
    r"""
    https://github.com/ai-bot-pro/achatbot/issues/133
    """

    TAG = "tts_orpheus"

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**OrpheusTTSArgs().__dict__, **kwargs}

    def __init__(self, **kwargs) -> None:
        self.args = OrpheusTTSArgs(**kwargs)
        self.args.lm_args = TransformersSpeechLMArgs(**self.args.lm_args)
        self.args.codec_args = CodecArgs(**self.args.codec_args)
        self.lm_model = TransformersManualSpeechLlama(**self.args.lm_args.__dict__)
        self.codec_model = SNACCodec(**self.args.codec_args.__dict__)

        self.voice_name = "tara"

    def get_stream_info(self) -> dict:
        return {
            # "format": PYAUDIO_PAINT16,
            "format": PYAUDIO_PAFLOAT32,
            "channels": 1,
            "rate": 24000,  # target_sample_rate
            "sample_width": 2,
            # "np_dtype": np.int16,
            "np_dtype": np.float32,
        }

    def set_voice(self, ref_file: str, **kwargs):
        logging.info(f"now, un support voice clone ref")

    def get_voices(self):
        return [
            "tara",
            "jess",
            "leo",
            "leah",
            "dan",
            "mia",
            "zac",
            "zoe",
        ]

    def token2wav(self, code_list):
        layer_1 = []
        layer_2 = []
        layer_3 = []
        for i in range((len(code_list) + 1) // 7):
            layer_1.append(code_list[7 * i])
            layer_2.append(code_list[7 * i + 1] - 4096)
            layer_3.append(code_list[7 * i + 2] - (2 * 4096))
            layer_3.append(code_list[7 * i + 3] - (3 * 4096))
            layer_2.append(code_list[7 * i + 4] - (4 * 4096))
            layer_3.append(code_list[7 * i + 5] - (5 * 4096))
            layer_3.append(code_list[7 * i + 6] - (6 * 4096))
        codes = [
            torch.tensor(layer_1, device=self.codec_model.device).unsqueeze(0),
            torch.tensor(layer_2, device=self.codec_model.device).unsqueeze(0),
            torch.tensor(layer_3, device=self.codec_model.device).unsqueeze(0),
        ]
        audio_hat = self.codec_model.decode_code(codes)
        return audio_hat.detach().cpu().numpy()

    def turn_token_id_to_id(self, token_id):
        """
        turn token id (lm tokenizer encode id) to vq indices
        128256: <custom_token_0>
        128266: <custom_token_10>
        """
        if token_id > (128256 + 10):
            return token_id - 128266
        return None

    def turn_token_to_id(self, token_string):
        """
        turn token (<custom_token_1234>) to vq indices 1234 - 10
        """
        # Strip whitespace
        token_string = token_string.strip()

        # Find the last token in the string
        last_token_start = token_string.rfind("<custom_token_")

        if last_token_start == -1:
            print("No token found in the string")
            return None

        # Extract the last token
        last_token = token_string[last_token_start:]

        # Process the last token
        if last_token.startswith("<custom_token_") and last_token.endswith(">"):
            try:
                number_str = last_token[14:-1]
                return int(number_str) - 10
            except ValueError:
                return None
        else:
            return None

    async def _inference(
        self, session: Session, text: str, **kwargs
    ) -> AsyncGenerator[bytes, None]:
        if "cuda" in str(self.lm_model._model.device):
            torch.cuda.empty_cache()
        seed = kwargs.get("seed", self.lm_args.lm_gen_seed)
        set_all_random_seed(seed)

        session.ctx.state["prompt"] = f"{self.voice_name}: {text}"
        streamer = self.lm_model.generate(session, **kwargs)

        buffer = []
        count = 0
        for token_id in streamer:
            # token = self.lm_model._tokenizer.decode(token_id)
            # code_id = self.turn_token_to_id(token)
            code_id = self.turn_token_id_to_id(token_id)
            if code_id is not None:
                buffer.append(code_id)
                count += 1
                if count % 7 == 0 and count >= 28:  # 28 tokens per sample
                    buffer_to_proc = buffer[-28:]  # slice 28 token ,7 tokens + overlap 21 tokens
                    audio_samples = self.token2wav(buffer_to_proc)
                    if audio_samples is not None:
                        yield audio_samples

            """
            - first < 28 tokens, no need to process
            - left < 7 tokens, no need to process
            """
            pass

        torch.cuda.empty_cache()
