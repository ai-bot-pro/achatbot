import logging
import random
import os
import sys
from typing import AsyncGenerator

from dotenv import load_dotenv

from src.common.interface import ITts
from src.common.session import Session
from src.common.types import CosyVoiceTTSArgs, RATE
from src.common.utils.audio_utils import postprocess_tts_wave_int16
from .base import BaseTTS

load_dotenv(override=True)


class CosyVoiceTTS(BaseTTS, ITts):
    r"""
    https://arxiv.org/abs/2407.05407v2
    """
    TAG = "tts_cosy_voice"

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**CosyVoiceTTSArgs().__dict__, **kwargs}

    def __init__(self, **args) -> None:
        if bool(os.getenv("ACHATBOT_PKG", "")):
            cur_dir = os.path.dirname(__file__)
            sys.path.insert(1, os.path.join(cur_dir, '../../../CosyVoice'))
            sys.path.insert(2, os.path.join(cur_dir, '../../../CosyVoice/third_party/Matcha-TTS'))
        else:
            sys.path.insert(1, os.path.join(sys.path[0], 'deps/CosyVoice'))
            sys.path.insert(2, os.path.join(sys.path[0], 'deps/CosyVoice/third_party/Matcha-TTS'))
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
        for name, model in {
            "llm_model": self.model.model.llm,
            "flow_model": self.model.model.flow,
            "hift_model": self.model.model.hift,
        }.items():
            model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
            logging.debug(f"{name} {model} {model_million_params}M parameters")

    def get_voices(self):
        spk_ids = []
        for spk_id in self.model.list_avaliable_spks():
            if self.args.language == "zh" and "中" in spk_id:
                spk_ids.append(spk_id)
            if self.args.language == "zh_yue" and "粤" in spk_id:
                spk_ids.append(spk_id)
            if self.args.language == "en" and "英" in spk_id:
                spk_ids.append(spk_id)
            if self.args.language == "jp" and "日" in spk_id:
                spk_ids.append(spk_id)
            if self.args.language == "ko" and "韩" in spk_id:
                spk_ids.append(spk_id)

        return spk_ids

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
        if 'tts_speech' in output:
            # torchaudio.save(os.path.join(RECORDS_DIR, f'{self.TAG}_speech.wav'),
            #                output['tts_speech'], 22050)
            res = postprocess_tts_wave_int16(output['tts_speech'])
            yield res
