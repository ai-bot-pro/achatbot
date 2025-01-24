import logging
import random
import os
import sys
from typing import AsyncGenerator

from dotenv import load_dotenv
import torchaudio

from .cosy_voice_tts import CosyVoiceTTS

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
    exit()

from src.common.interface import ITts
from src.common.session import Session
from src.common.types import RECORDS_DIR, CosyVoiceTTSArgs, RATE
from src.common.utils.audio_utils import postprocess_tts_wave_int16

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

        # CosyVoice2 no spk_ids, so need reference_audio
        self.set_voice(self.args.reference_audio_path)

        self.clear()
        self.log_parameters()

    def clear(self):
        """
        CosyVoice2 和 CosyVoice 的配置不同，
        cosyvoice: https://huggingface.co/FunAudioLLM/CosyVoice-300M/blob/main/cosyvoice.yaml
        cosyvoice2: https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B/blob/main/cosyvoice.yaml
        调完老版本CosyVoice，再调要CosyVoice2， 需要对hann.window和mel_basis重置一下
        """
        hann_window.clear()
        mel_basis.clear()

    def set_voice(self, reference_audio_path: str):
        if os.path.exists(reference_audio_path) is False:
            raise FileNotFoundError(f"reference_audio_path: {reference_audio_path}")

        self.args.reference_audio_path = reference_audio_path
        self.reference_audio = load_wav(self.args.reference_audio_path, RATE)

    def set_src_voice(self, src_audio_path: str):
        if os.path.exists(src_audio_path) is False:
            raise FileNotFoundError(f"src_audio_path: {src_audio_path}")

        self.args.src_audio_path = src_audio_path
        self.src_audio = load_wav(self.args.src_audio_path, RATE)

    def get_voices(self):
        if self.args.reference_audio_path:
            return [self.args.reference_audio_path]
        return []

    async def _inference(
        self, session: Session, text: str, **kwargs
    ) -> AsyncGenerator[bytes, None]:
        if len(self.args.instruct_text.strip()) > 0:
            # if have instruct_text then use instruct2 infer
            logging.debug("instruct2 infer")
            output = self.model.inference_instruct2(
                text,
                self.args.instruct_text,
                self.reference_audio,
                stream=self.args.tts_stream,
                speed=self.args.tts_speed,
            )
        else:
            if len(self.args.reference_text.strip()) > 0:
                # if have reference_text then use zero shot infer
                logging.debug("zero shot infer")
                output = self.model.inference_zero_shot(
                    text,
                    self.args.reference_text,
                    self.reference_audio,
                    stream=self.args.tts_stream,
                    speed=self.args.tts_speed,
                )
            else:
                # if no reference_text and no instruct_text,
                # default use cross lingual infer;
                # fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
                logging.debug("cross lingual infer")
                output = self.model.inference_cross_lingual(
                    text,
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
