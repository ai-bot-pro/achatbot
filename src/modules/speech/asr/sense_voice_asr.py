from typing import AsyncGenerator
import asyncio
import re

from src.common.device_cuda import CUDAInfo
from src.common.utils.audio_utils import bytes2TorchTensorWith16
from src.common.session import Session
from src.modules.speech.asr.base import ASRBase


class SenseVoiceAsr(ASRBase):
    TAG = "sense_voice_asr"

    def __init__(self, **args) -> None:
        from deps.SenseVoice.model import SenseVoiceSmall

        super().__init__(**args)
        device = "cpu"
        if self.args.device:
            device = self.args.device
        else:
            info = CUDAInfo()
            if info.is_cuda:
                device = "cuda:0"
        self.model: SenseVoiceSmall = None
        self.model, self.kwargs = SenseVoiceSmall.from_pretrained(
            model=self.args.model_name_or_path, device=device
        )
        self.model.eval()

    def set_audio_data(self, audio_data):
        if isinstance(audio_data, (bytes, bytearray)):
            self.asr_audio = bytes2TorchTensorWith16(audio_data)
        if isinstance(audio_data, str):
            self.asr_audio = audio_data
        return

    async def transcribe_stream(self, session: Session) -> AsyncGenerator[str, None]:
        transcription, _ = await asyncio.to_thread(
            self.model.inference,
            data_in=self.asr_audio,
            language=self.args.language,  # "zn", "en", "yue", "ja", "ko", "nospeech"
            use_itn=False,  # use Inverse Text Normalization，ITN
            **self.kwargs,
        )
        for item in transcription:
            clean_text = re.sub(r"<\|.*?\|>", "", item["text"])
            yield clean_text

    async def transcribe(self, session: Session) -> dict:
        transcription, meta_data = await asyncio.to_thread(
            self.model.inference,
            data_in=self.asr_audio,
            language=self.args.language,  # "zn", "en", "yue", "ja", "ko", "nospeech"
            use_itn=False,  # use Inverse Text Normalization，ITN
            ban_emo_unk=False,
            **self.kwargs,
        )
        clean_text = re.sub(r"<\|.*?\|>", "", transcription[0]["text"])
        res = {
            "language": self.args.language,
            "language_probability": None,
            "text": clean_text,
            "words": [],
        }
        return res
