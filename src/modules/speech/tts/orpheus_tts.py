import logging
import math
from typing import AsyncGenerator

from dotenv import load_dotenv
import numpy as np
import torch

from src.core.llm.transformers.manual_speech_orpheus import TransformersManualSpeechOrpheus
from src.common.random import set_all_random_seed
from src.common.interface import ITts
from src.common.session import Session
from src.common.types import PYAUDIO_PAFLOAT32, PYAUDIO_PAINT16
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
        self.lm_model = TransformersManualSpeechOrpheus(**self.args.lm_args.__dict__)
        self.codec_model = SNACCodec(**self.args.codec_args.__dict__)

        self.voice_name = self.args.voice_name

    def get_stream_info(self) -> dict:
        return {
            "format": PYAUDIO_PAINT16,
            # "format": PYAUDIO_PAFLOAT32,
            "channels": 1,
            "rate": 24000,  # target_sample_rate
            "sample_width": 2,
            "np_dtype": np.int16,
            # "np_dtype": np.float32,
        }

    def set_voice(self, ref_file: str, **kwargs):
        logging.info(f"now, un support voice clone ref")
        self.voice_name = ref_file

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
            torch.tensor(layer_1, device=self.codec_model.args.device).unsqueeze(0),
            torch.tensor(layer_2, device=self.codec_model.args.device).unsqueeze(0),
            torch.tensor(layer_3, device=self.codec_model.args.device).unsqueeze(0),
        ]
        audio_hat = self.codec_model.decode_code(codes)

        audio_np = audio_hat.detach().cpu().numpy()
        audio_np = (audio_np * 32767).astype(np.int16)
        return audio_np

    def turn_token_id_to_id(self, token_id):
        """
        turn token id (lm tokenizer encode id) to vq indices
        128256: <custom_token_0>
        128266: <custom_token_10>
        """
        if token_id > (128256 + 10):
            return token_id - 128266
        return None

    async def _inference(
        self, session: Session, text: str, **kwargs
    ) -> AsyncGenerator[bytes, None]:
        if "cuda" in str(self.lm_model._model.device):
            torch.cuda.empty_cache()
        seed = kwargs.get("seed", self.args.lm_args.lm_gen_seed)
        set_all_random_seed(seed)

        session.ctx.state["prompt"] = f"{self.voice_name}: {text}"
        streamer = self.lm_model.generate(session, **kwargs)

        stream_factor = kwargs.get("stream_factor", self.args.stream_factor)
        token_overlap_len = kwargs.get("token_overlap_len", self.args.token_overlap_len)

        chunk_size = math.ceil(stream_factor * 28)
        logging.info(f"init chunk_size: {chunk_size} token_overlap_len:{token_overlap_len}")

        semantic_token_ids = []
        for token_id in streamer:
            code_id = self.turn_token_id_to_id(token_id)
            if code_id is not None:
                semantic_token_ids.append(code_id)
                # if len(semantic_token_ids) % chunk_size == 0:
                if len(semantic_token_ids) >= chunk_size + token_overlap_len:
                    waveform = self.token2wav(semantic_token_ids)
                    if waveform is not None:
                        yield waveform.tobytes()

                    semantic_token_ids = semantic_token_ids[chunk_size:]

        if len(semantic_token_ids) > 0:  # end to finalize
            waveform = self.token2wav(semantic_token_ids)
            if waveform is not None:
                yield waveform.tobytes()
            logging.info(f"last chunk len: {len(semantic_token_ids)}")

        torch.cuda.empty_cache()
