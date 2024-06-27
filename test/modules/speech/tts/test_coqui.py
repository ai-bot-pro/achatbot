import os
import logging
import asyncio

import unittest
import pyaudio

from src.common.utils.audio_utils import save_audio_to_file
from src.common.factory import EngineFactory
from src.common.logger import Logger
from src.common.session import Session
from src.common.types import SessionCtx, MODELS_DIR, RECORDS_DIR
from src.modules.speech.tts.coqui_tts import EngineClass, CoquiTTS

r"""
REFERENCE_AUDIO_PATH= python -m unittest test.modules.speech.tts.test_coqui.TestCoquiTTS.test_synthesize
python -m unittest test.modules.speech.tts.test_coqui.TestCoquiTTS.test_synthesize
STREAM=1 python -m unittest test.modules.speech.tts.test_coqui.TestCoquiTTS.test_synthesize
"""


class TestCoquiTTS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tts_tag = os.getenv('TTS_TAG', "tts_coqui")
        cls.tts_text = os.getenv(
            'TTS_TEXT', "你好！有什么可以帮助你的吗？请描述一下您的问题或需求。")
        cls.stream = os.getenv('STREAM', "")
        cls.conf_file = os.getenv(
            'CONF_FILE', os.path.join(MODELS_DIR, "coqui/XTTS-v2/config.json"))
        cls.model_path = os.getenv('MODEL_PATH', os.path.join(
            MODELS_DIR, "coqui/XTTS-v2"))
        cls.reference_audio_path = os.getenv('REFERENCE_AUDIO_PATH', os.path.join(
            RECORDS_DIR, "tmp.wav"))
        Logger.init(logging.DEBUG, is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        kwargs = {}
        kwargs["model_path"] = self.model_path
        kwargs["conf_file"] = self.conf_file
        kwargs["reference_audio_path"] = self.reference_audio_path
        self.tts: CoquiTTS = EngineFactory.get_engine_by_tag(
            EngineClass, self.tts_tag, **kwargs)
        self.session = Session(**SessionCtx("test_tts_client_id").__dict__)

        stream_info = self.tts.get_stream_info()
        self.pyaudio_instance = pyaudio.PyAudio()
        self.audio_stream = self.pyaudio_instance.open(
            format=stream_info["format_"],
            channels=stream_info["channels"],
            rate=stream_info["rate"],
            output_device_index=None,
            output=True)

    def tearDown(self):
        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.pyaudio_instance.terminate()

    def test_synthesize(self):
        self.session.ctx.state["tts_text"] = self.tts_text
        print(self.session.ctx)
        self.tts.args.tts_stream = bool(self.stream)
        iter = self.tts.synthesize_sync(self.session)
        sub_chunk_size = 1024
        for i, chunk in enumerate(iter):
            print(f"get {i} chunk {len(chunk)}")
            self.assertGreater(len(chunk), 0)
            if len(chunk) / sub_chunk_size < 100:
                self.audio_stream.write(chunk)
                continue
            for i in range(0, len(chunk), sub_chunk_size):
                sub_chunk = chunk[i:i + sub_chunk_size]
                self.audio_stream.write(sub_chunk)

    def test_get_voices(self):
        voices = self.tts.get_voices()
        self.assertGreater(len(voices), 0)
        print(voices, len(voices))
