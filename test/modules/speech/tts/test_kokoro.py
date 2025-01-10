import os
import logging

import random
import unittest

from src.modules.speech.tts import TTSEnvInit
from src.common.interface import ITts
from src.common.factory import EngineClass
from src.common.logger import Logger
from src.common.session import Session
from src.common.types import SessionCtx

r"""
python -m unittest test.modules.speech.tts.test_kokoro.TestKOKORO.test_get_voices
python -m unittest test.modules.speech.tts.test_kokoro.TestKOKORO.test_set_voice
python -m unittest test.modules.speech.tts.test_kokoro.TestKOKORO.test_synthesize

TTS_TAG=tts_onnx_kokoro python -m unittest test.modules.speech.tts.test_kokoro.TestKOKORO.test_get_voices
TTS_TAG=tts_onnx_kokoro python -m unittest test.modules.speech.tts.test_kokoro.TestKOKORO.test_set_voice
TTS_TAG=tts_onnx_kokoro KOKORO_ESPEAK_NG_LIB_PATH=/usr/local/lib/libespeak-ng.1.dylib python -m unittest test.modules.speech.tts.test_kokoro.TestKOKORO.test_synthesize
TTS_STREAM=1 TTS_TAG=tts_onnx_kokoro KOKORO_ESPEAK_NG_LIB_PATH=/usr/local/lib/libespeak-ng.1.dylib python -m unittest test.modules.speech.tts.test_kokoro.TestKOKORO.test_synthesize
"""


class TestKOKORO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tts_tag = os.getenv("TTS_TAG", "tts_kokoro")
        cls.tts_text = os.getenv(
            "TTS_TEXT",
            # "hello, test.modules.speech.tts.test_kokoro.TestKOKORO.test_synthesize.",
            "你好，我是机器人",
            # "How could I know? It's an unanswerable question. Like asking an unborn child if they'll lead a good life. They haven't even been born.",
        )
        Logger.init(os.getenv("LOG_LEVEL", "debug").upper(), is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.tts: EngineClass | ITts = TTSEnvInit.initTTSEngine(self.tts_tag)
        self.session = Session(**SessionCtx("test_tts_client_id").__dict__)

    def tearDown(self):
        pass

    def test_get_voices(self):
        voices = self.tts.get_voices()
        self.assertGreater(len(voices), 0)
        print(voices)

    def test_set_voice(self):
        voice = random.choice(self.tts.get_voices())
        self.tts.set_voice(voice)

        print(voice)
        self.assertEqual(self.tts.args.voice, voice)

    def test_synthesize(self):
        import pyaudio

        stream_info = self.tts.get_stream_info()
        print(stream_info)
        self.pyaudio_instance = pyaudio.PyAudio()
        self.audio_stream = self.pyaudio_instance.open(
            format=stream_info["format"],
            channels=stream_info["channels"],
            rate=stream_info["rate"],
            output_device_index=None,
            output=True,
        )

        self.test_set_voice()

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
