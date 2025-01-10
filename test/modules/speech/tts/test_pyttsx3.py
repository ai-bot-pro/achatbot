import os
import logging

import unittest

from src.common.factory import EngineFactory, EngineClass
from src.common.logger import Logger
from src.common.session import Session
from src.common.types import SessionCtx
from src.modules.speech.tts.pyttsx3_tts import Pyttsx3TTS

r"""
python -m unittest test.modules.speech.tts.test_pyttsx3.TestPyttsx3TTS.test_get_voices
python -m unittest test.modules.speech.tts.test_pyttsx3.TestPyttsx3TTS.test_synthesize
"""


class TestPyttsx3TTS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tts_tag = os.getenv("LLM_TAG", "tts_pyttsx3")
        cls.tts_text = os.getenv("TTS_TEXT", "你好，我是机器人")
        cls.voice_name = os.getenv("VOICE_NAME", "Tingting")
        Logger.init(os.getenv("LOG_LEVEL", "debug").upper(), is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        kwargs = {}
        kwargs["voice_name"] = self.voice_name
        self.tts: Pyttsx3TTS = EngineFactory.get_engine_by_tag(EngineClass, self.tts_tag, **kwargs)
        self.session = Session(**SessionCtx("test_tts_client_id").__dict__)

    def tearDown(self):
        pass

    def test_get_voices(self):
        voices = self.tts.get_voices()
        self.assertGreater(len(voices), 0)
        print(voices)

    def test_synthesize(self):
        import pyaudio

        stream_info = self.tts.get_stream_info()
        self.pyaudio_instance = pyaudio.PyAudio()
        self.audio_stream = self.pyaudio_instance.open(
            format=stream_info["format"],
            channels=stream_info["channels"],
            rate=stream_info["rate"],
            output_device_index=None,
            output=True,
        )

        self.session.ctx.state["tts_text"] = self.tts_text
        print(self.session.ctx)
        iter = self.tts.synthesize_sync(self.session)
        sub_chunk_size = 1024
        for i, chunk in enumerate(iter):
            print(f"get {i} chunk {len(chunk)}")
            self.assertGreaterEqual(len(chunk), 0)
            if len(chunk) / sub_chunk_size < 100:
                self.audio_stream.write(chunk)
                continue
            for i in range(0, len(chunk), sub_chunk_size):
                sub_chunk = chunk[i : i + sub_chunk_size]
                self.audio_stream.write(sub_chunk)

        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.pyaudio_instance.terminate()
