import logging
from typing import AsyncGenerator, Iterator, Union
import os

from pydub.utils import mediainfo
from pydub import AudioSegment

from src.common.utils.wav import read_audio_file
from src.common.interface import ITts
from src.common.session import Session
from src.common.types import Pyttsx3TTSArgs, PYTTSX3_SYNTHESIS_FILE, RECORDS_DIR
from .base import BaseTTS, EngineClass, TTSVoice


class Pyttsx3TTS(BaseTTS, ITts):
    TAG = "tts_pyttsx3"

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**Pyttsx3TTSArgs().__dict__, **kwargs}

    def __init__(self, **args) -> None:
        import pyttsx3

        self.args = Pyttsx3TTSArgs(**args)
        self.engine = pyttsx3.init()
        self.set_voice(self.args.voice_name)
        self.file_path = os.path.join(RECORDS_DIR, PYTTSX3_SYNTHESIS_FILE)

    async def _inference(
        self, session: Session, text: str, **kwargs
    ) -> AsyncGenerator[bytes, None]:
        logging.debug(f"{self.TAG} synthesis: {text} save to file: {self.file_path}")
        self.engine.save_to_file(text, self.file_path)
        self.engine.runAndWait()

        # Get media info of the file
        info = mediainfo(self.file_path)
        logging.debug(f"{self.file_path} media info: {info}")

        # Check if the file format is AIFF and convert to WAV if necessary
        if info["format_name"] == "aiff":
            audio = AudioSegment.from_(self.file_path, format="aiff")
            audio.export(self.file_path, format="wav")

        audio_data = await read_audio_file(self.file_path)
        yield audio_data

    def get_voices(self):
        voice_objects = []
        voices = self.engine.getProperty("voices")
        for voice in voices:
            voice_objects.append(voice.name)
        return voice_objects

    def set_voice(self, voice: Union[str, TTSVoice]):
        if isinstance(voice, TTSVoice):
            self.engine.setProperty("voice", voice.id)
        else:
            installed_voices = self.engine.getProperty("voices")
            if voice is not None:
                for installed_voice in installed_voices:
                    if voice in installed_voice.name:
                        logging.debug(
                            f"{self.TAG} set voice: {voice} voice_id: {installed_voice.id}"
                        )
                        self.engine.setProperty("voice", installed_voice.id)
                        return

    def set_voice_parameters(self, **voice_parameters):
        for parameter, value in voice_parameters.items():
            self.engine.setProperty(parameter, value)
