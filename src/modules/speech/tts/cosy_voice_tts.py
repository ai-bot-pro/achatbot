import logging
import random
import os
import sys
from typing import AsyncGenerator

from dotenv import load_dotenv
import numpy as np
import torchaudio

from src.common.logger import Logger

try:
    cur_dir = os.path.dirname(__file__)
    if bool(os.getenv("ACHATBOT_PKG", "")):
        sys.path.insert(1, os.path.join(cur_dir, "../../../CosyVoice"))
        sys.path.insert(2, os.path.join(cur_dir, "../../../CosyVoice/third_party/Matcha-TTS"))
    else:
        sys.path.insert(1, os.path.join(cur_dir, "../../../../deps/CosyVoice"))
        sys.path.insert(
            2, os.path.join(cur_dir, "../../../../deps/CosyVoice/third_party/Matcha-TTS")
        )
    from deps.CosyVoice.cosyvoice.cli.cosyvoice import CosyVoice
    from deps.CosyVoice.cosyvoice.utils.file_utils import load_wav, logging
    # logging.shutdown()
except ImportError:
    logging.error("Failed to import CosyVoice, need pip install achatbot[tts_cosey_voice] ")
    exit()

from src.common.interface import ITts
from src.common.session import Session
from src.common.types import PYAUDIO_PAINT16, RECORDS_DIR, CosyVoiceTTSArgs, RATE
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
        self.args = CosyVoiceTTSArgs(**args)
        self.model = CosyVoice(self.args.model_dir)
        voices = self.get_voices()
        if len(voices) > 0 and self.args.spk_id not in voices:
            self.args.spk_id = random.choice(voices)
        self.src_audio = None
        if os.path.exists(self.args.src_audio_path) is True:
            self.src_audio = load_wav(self.args.src_audio_path, RATE)
        self.reference_audio = None
        if os.path.exists(self.args.reference_audio_path) is True:
            self.reference_audio = load_wav(self.args.reference_audio_path, RATE)

        self.log_parameters()

    def log_parameters(self):
        total_parameters = 0
        for name, model in {
            "llm_model": self.model.model.llm,
            "flow_model": self.model.model.flow,
            "hift_model": self.model.model.hift,
        }.items():
            params = sum(p.numel() for p in model.parameters())
            total_parameters += params
            model_million_params = params / 1e6
            logging.debug(f"{name} {model} {model_million_params}M parameters")
        logging.debug(f"total {total_parameters / 1e6}M parameters")

    def get_voices(self):
        spk_ids = []
        for spk_id in self.model.list_available_spks():
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

    def get_stream_info(self) -> dict:
        return {
            "format": PYAUDIO_PAINT16,
            # "format": PYAUDIO_PAFLOAT32,
            "channels": 1,
            "rate": 22050,  # target_sample_rate
            "sample_width": 2,
            "np_dtype": np.int16,
            # "np_dtype": np.float32,
        }

    def set_voice(self, spk_id: str):
        # now just support spk_id
        if spk_id in self.model.list_available_spks():
            self.args.spk_id = spk_id
        else:
            logging.warning(f"Speaker ID {spk_id} not found! Using speaker ID {self.args.spk_id}")

    def filter_special_chars(self, text: str) -> str:
        # @TODO: use nlp stream process sentence
        special_chars = ".。,，;；!！?？」>》}\\】\"”'‘)~"
        return self._filter_special_chars(special_chars, text)

    async def _inference(
        self, session: Session, text: str, **kwargs
    ) -> AsyncGenerator[bytes, None]:
        if self.src_audio is not None and self.reference_audio is not None:
            logging.debug("vc(voice convert) infer")
            output = self.model.inference_vc(
                self.src_audio,
                self.reference_audio,
                stream=self.args.tts_stream,
                speed=self.args.tts_speed,
            )
        else:
            if self.reference_audio is None:
                if len(self.args.instruct_text.strip()) == 0:
                    logging.debug("sft(speaker fine tune) infer")
                    output = self.model.inference_sft(
                        text,
                        self.args.spk_id,
                        stream=self.args.tts_stream,
                        speed=self.args.tts_speed,
                    )
                else:
                    logging.debug("instruct infer")
                    output = self.model.inference_instruct(
                        text,
                        self.args.spk_id,
                        self.args.instruct_text,
                        stream=self.args.tts_stream,
                        speed=self.args.tts_speed,
                    )
            else:
                if len(self.args.reference_text.strip()) == 0:
                    logging.debug("cross lingual infer")
                    output = self.model.inference_cross_lingual(
                        text,
                        self.reference_audio,
                        stream=self.args.tts_stream,
                        speed=self.args.tts_speed,
                    )
                else:
                    logging.debug("zero shot infer")
                    output = self.model.inference_zero_shot(
                        text,
                        self.args.reference_text,
                        self.reference_audio,
                        stream=self.args.tts_stream,
                        speed=self.args.tts_speed,
                    )

        for item in output:
            if "tts_speech" in item:
                # torchaudio.save(
                #    os.path.join(RECORDS_DIR, f"{self.TAG}_speech.wav"), item["tts_speech"], 22050
                # )
                res = postprocess_tts_wave_int16(item["tts_speech"])
                yield res
