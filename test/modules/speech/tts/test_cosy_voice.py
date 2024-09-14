import os
import logging
import asyncio

import unittest

from src.modules.speech.tts.cosy_voice_tts import CosyVoiceTTS
from src.common.factory import EngineFactory, EngineClass
from src.common.logger import Logger
from src.common.session import Session
from src.common.utils.wav import save_audio_to_file
from src.common.types import SessionCtx, CosyVoiceTTSArgs, MODELS_DIR

r"""
python -m unittest test.modules.speech.tts.test_cosy_voice.TestCosyVoiceTTS.test_synthesize
python -m unittest test.modules.speech.tts.test_cosy_voice.TestCosyVoiceTTS.test_get_voices
"""


class TestCosyVoiceTTS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tts_tag = os.getenv('TTS_TAG', "tts_cosy_voice")
        cls.tts_text = os.getenv(
            'TTS_TEXT',
            "你好，我是机器人, hello, test.modules.speech.tts.test_gtts.TestGTTS.test_synthesize")
        model_dir = os.path.join(MODELS_DIR, "CosyVoice-300M-SFT")
        cls.model_dir = os.getenv('MODELS_DIR', model_dir)
        Logger.init(os.getenv("LOG_LEVEL", "debug").upper(), is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        kwargs = CosyVoiceTTSArgs(
            model_dir=self.model_dir,
            reference_audio_path=os.getenv('REFERENCE_AUDIO_PATH', ''),
            instruct_text=os.getenv('INSTRUCT_TEXT', ''),
            spk_id=os.getenv('SPK_ID', ""),
        ).__dict__
        self.tts: CosyVoiceTTS = EngineFactory.get_engine_by_tag(
            EngineClass, self.tts_tag, **kwargs)
        self.session = Session(**SessionCtx("test_tts_client_id").__dict__)
        self.pyaudio_instance = None
        self.audio_stream = None

    def tearDown(self):
        self.audio_stream and self.audio_stream.stop_stream()
        self.audio_stream and self.audio_stream.close()
        self.pyaudio_instance and self.pyaudio_instance.terminate()

    def test_get_voices(self):
        voices = self.tts.get_voices()
        self.assertGreater(len(voices), 0)
        print(voices)

    def test_synthesize(self):
        stream_info = self.tts.get_stream_info()
        self.session.ctx.state["tts_text"] = self.tts_text
        print(self.session.ctx)
        iter = self.tts.synthesize_sync(self.session)
        res = bytearray()
        for i, chunk in enumerate(iter):
            print(i, len(chunk))
            res.extend(chunk)
        path = asyncio.run(save_audio_to_file(
            res,
            f"test_{self.tts.TAG}.wav",
            sample_rate=stream_info["rate"],
            sample_width=stream_info["sample_width"],
            channles=stream_info["channels"],
        ))
        print(path)

    def test_synthesize_speak(self):
        import pyaudio
        stream_info = self.tts.get_stream_info()
        self.pyaudio_instance = pyaudio.PyAudio()
        self.audio_stream = self.pyaudio_instance.open(
            format=stream_info["format"],
            channels=stream_info["channels"],
            rate=stream_info["rate"],
            output_device_index=None,
            output=True)

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
                sub_chunk = chunk[i:i + sub_chunk_size]
                self.audio_stream.write(sub_chunk)
