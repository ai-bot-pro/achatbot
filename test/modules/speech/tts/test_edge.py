import os
import logging
import asyncio

import unittest

from src.common.factory import EngineFactory, EngineClass
from src.common.logger import Logger
from src.common.session import Session
from src.common.types import SessionCtx
from src.modules.speech.tts.edge_tts import EdgeTTS

r"""
python -m unittest test.modules.speech.tts.test_edge.TestEdgeTTS.test_get_voices
python -m unittest test.modules.speech.tts.test_edge.TestEdgeTTS.test_synthesize
"""


class TestEdgeTTS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tts_tag = os.getenv("TTS_TAG", "tts_edge")
        cls.tts_text = os.getenv(
            "TTS_TEXT",
            "hello, 你好，我是机器人, 有什么可以帮助您的吗？请告诉我您需要的信息或问题，我会尽力为您解答。",
        )
        Logger.init(os.getenv("LOG_LEVEL", "debug").upper(), is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        kwargs = {}
        kwargs["voice_name"] = "zh-CN-XiaoxiaoNeural"
        self.tts: EdgeTTS = EngineFactory.get_engine_by_tag(EngineClass, self.tts_tag, **kwargs)
        self.session = Session(**SessionCtx("test_tts_client_id").__dict__)

    def tearDown(self):
        pass

    def test_get_voices(self):
        voices = self.tts.get_voices()
        self.assertGreater(len(voices), 0)
        print(voices, len(voices))

        voices = asyncio.run(self.tts._get_voices(Language="zh", Gender="Female"))
        self.assertGreater(len(voices), 0)
        print(voices, len(voices))

        voices = asyncio.run(self.tts._get_voices(Language="zh", Gender="Male"))
        self.assertGreater(len(voices), 0)
        print(voices, len(voices))

        voices = asyncio.run(self.tts._get_voices(Language="en", Gender="Female"))
        self.assertGreater(len(voices), 0)
        print(voices, len(voices))

        voices = asyncio.run(self.tts._get_voices(Language="en", Gender="Male"))
        self.assertGreater(len(voices), 0)
        print(voices, len(voices))

        voices = asyncio.run(self.tts._get_voices(ShortName=self.tts.args.voice_name))
        self.assertGreater(len(voices), 0)
        print(voices, len(voices))

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

    def test_inference(self) -> None:
        async def inference():
            async for chunk in self.tts._inference(self.session, self.tts_text):
                self.assertGreaterEqual(len(chunk), 0)

        asyncio.run(inference())
