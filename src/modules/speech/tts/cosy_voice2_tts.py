import logging
import random
import os
import sys
from typing import AsyncGenerator

from dotenv import load_dotenv

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
    from deps.CosyVoice.cosyvoice.cli.cosyvoice import CosyVoice2
    from deps.CosyVoice.cosyvoice.utils.file_utils import load_wav
    from matcha.utils.audio import hann_window, mel_basis

except ImportError:
    logging.error("Failed to import CosyVoice2, need pip install achatbot[tts_cosey_voice2] ")
    pass

from src.common.interface import ITts
from src.common.session import Session
from src.common.types import CosyVoiceTTSArgs, RATE
from src.common.utils.audio_utils import postprocess_tts_wave_int16
from .cosy_voice_tts import CosyVoiceTTS

load_dotenv(override=True)


class CosyVoice2TTS(CosyVoiceTTS):
    r"""
    https://arxiv.org/abs/2412.10117
    """

    TAG = "tts_cosy_voice2"

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**CosyVoiceTTSArgs().__dict__, **kwargs}

    def __init__(self, **args) -> None:
        self.args = CosyVoiceTTSArgs(**args)
        self.model = CosyVoice2(self.args.model_dir)
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

    def clear(self):
        # CosyVoice2 和 CosyVoice 的配置不同，
        # cosyvoice: https://huggingface.co/FunAudioLLM/CosyVoice-300M/blob/main/cosyvoice.yaml
        # cosyvoice2: https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B/blob/main/cosyvoice.yaml
        # 调完老版本CosyVoice，再调要CosyVoice2， 需要对hann.window和mel_basis重置一下
        hann_window.clear()
        mel_basis.clear()

    async def _inference(
        self, session: Session, text: str, **kwargs
    ) -> AsyncGenerator[bytes, None]:
        if self.reference_audio is None:
            if len(self.args.instruct_text.strip()) == 0:
                output = self.model.inference_sft(
                    text, self.args.spk_id, stream=self.args.tts_stream, speed=self.args.tts_speed
                )
            else:
                output = self.model.inference_instruct2(
                    text,
                    self.args.instruct_text,
                    stream=self.args.tts_stream,
                    speed=self.args.tts_speed,
                )
        else:
            if len(self.args.reference_text.strip()) == 0:
                output = self.model.inference_cross_lingual(
                    text,
                    self.reference_audio,
                    stream=self.args.tts_stream,
                    speed=self.args.tts_speed,
                )
            else:
                output = self.model.inference_zero_shot(
                    text,
                    self.args.reference_text,
                    self.reference_audio,
                    stream=self.args.tts_stream,
                    speed=self.args.tts_speed,
                )
        if "tts_speech" in output:
            # torchaudio.save(os.path.join(RECORDS_DIR, f'{self.TAG}_speech.wav'),
            #                output['tts_speech'], 22050)
            res = postprocess_tts_wave_int16(output["tts_speech"])
            yield res
