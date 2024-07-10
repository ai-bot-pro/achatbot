import logging
import random
import os
from typing import AsyncGenerator


from src.common.interface import ITts
from src.common.session import Session
from src.common.types import CosyVoiceTTSArgs, RATE
from src.common.utils.audio_utils import postprocess_tts_wave
from .base import BaseTTS


class CosyVoiceTTS(BaseTTS, ITts):
    r"""
    https://arxiv.org/abs/2407.05407v2
    """
    TAG = "tts_cosy_voice"

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**CosyVoiceTTSArgs().__dict__, **kwargs}

    def __init__(self, **args) -> None:
        from deps.CosyVoice.cosyvoice.cli.cosyvoice import CosyVoice
        from deps.CosyVoice.cosyvoice.utils.file_utils import load_wav
        self.args = CosyVoiceTTSArgs(**args)
        self.model = CosyVoice(self.args.model_dir)
        voices = self.get_voices()
        if self.args.spk_id not in voices:
            self.args.spk_id = random.choice(voices)
        self.reference_audio = None
        if os.path.exists(self.args.reference_audio_path):
            self.reference_audio = load_wav(self.args.reference_audio_path, RATE)

    def get_voices(self):
        return self.model.list_avaliable_spks()

    async def _inference(self, session: Session, text: str) -> AsyncGenerator[bytes, None]:
        if self.reference_audio is None:
            if len(self.args.instruct_text.strip()) == 0:
                output = self.model.inference_sft(text, self.args.spk_id)
            else:
                output = self.model.inference_instruct(
                    text, self.args.spk_id, self.args.instruct_text)
        else:
            if len(self.args.instruct_text.strip()) == 0:
                output = self.model.inference_cross_lingual(text, self.reference_audio)
            else:
                output = self.model.inference_zero_shot(
                    text, self.args.instruct_text, self.reference_audio)
        if hasattr(output, 'tts_speech'):
            res = postprocess_tts_wave(output['tts_speech'])
            yield res
