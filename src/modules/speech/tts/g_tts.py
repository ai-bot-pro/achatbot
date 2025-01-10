import logging
from typing import AsyncGenerator, Iterator, Union
import os
import io

from pydub import AudioSegment

from src.common.interface import ITts
from src.common.session import Session
from src.common.types import GTTS_SYNTHESIS_FILE, GTTSArgs, RECORDS_DIR
from .base import BaseTTS


class GTTS(BaseTTS, ITts):
    TAG = "tts_g"

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**GTTSArgs().__dict__, **kwargs}

    def __init__(self, **args) -> None:
        self.args = GTTSArgs(**args)
        self.file_path = os.path.join(RECORDS_DIR, GTTS_SYNTHESIS_FILE)

    async def _inference(
        self, session: Session, text: str, **kwargs
    ) -> AsyncGenerator[bytes, None]:
        from gtts import gTTS

        with io.BytesIO() as f:
            tts = gTTS(text=text, lang=self.args.language, tld=self.args.tld, slow=self.args.slow)
            tts.write_to_fp(f)
            f.seek(0)

            audio: AudioSegment = AudioSegment.from_mp3(f)
            if self.args.speed_increase != 1.0:
                audio = audio.speedup(
                    playback_speed=self.args.speed_increase,
                    chunk_size=self.args.chunk_size,
                    crossfade=self.args.crossfade_lenght,
                )
            audio_resampled = (
                audio.set_frame_rate(22050).set_channels(1).set_sample_width(2)
            )  # 16bit sample_width 16/8=2
            audio_data = audio_resampled.raw_data
            yield audio_data

    def get_voices(self):
        from gtts.lang import tts_langs

        voices = []
        languages = tts_langs()
        tlds = ["com", "com.au", "co.uk", "us", "ca", "co.in", "ie", "co.za"]

        for lang in languages.keys():
            for tld in tlds:
                voices.append(f"{lang}-{tld}")

        return voices
