import logging
from typing import Iterator, Union
import os

import wave
import pyaudio
from pydub.utils import mediainfo
from pydub import AudioSegment

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

    def get_stream_info(self) -> dict:
        return {
            "format_": pyaudio.paInt16,
            "channels": 1,
            "rate": 22050,
        }

    def _inference(self, session: Session, text: str) -> Iterator[bytes]:
        logging.debug(
            f"{self.TAG} synthesis: {text} save to file: {self.file_path}")
        self.engine.save_to_file(text, self.file_path)
        self.engine.runAndWait()

        # Get media info of the file
        info = mediainfo(self.file_path)
        logging.debug(f"{self.file_path} media info: {info}")

        # Check if the file format is AIFF and convert to WAV if necessary
        if info['format_name'] == 'aiff':
            audio = AudioSegment.from_file(self.file_path, format="aiff")
            audio.export(self.file_path, format="wav")

        # Open the WAV file
        with wave.open(self.file_path, 'rb') as wf:
            audio_data = wf.readframes(wf.getnframes())
            yield audio_data

    def get_voices(self):
        voice_objects = []
        voices = self.engine.getProperty('voices')
        for voice in voices:
            voice_object = TTSVoice(voice.name, voice.id)
            voice_objects.append(voice_object)
        return voice_objects

    def set_voice(self, voice: Union[str, TTSVoice]):
        if isinstance(voice, TTSVoice):
            self.engine.setProperty('voice', voice.id)
        else:
            installed_voices = self.engine.getProperty('voices')
            if not voice is None:
                for installed_voice in installed_voices:
                    if voice in installed_voice.name:
                        self.engine.setProperty('voice', installed_voice.id)
                        return

    def set_voice_parameters(self, **voice_parameters):
        for parameter, value in voice_parameters.items():
            self.engine.setProperty(parameter, value)
