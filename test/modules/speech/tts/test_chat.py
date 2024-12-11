import os
import logging
import asyncio

import unittest

from src.common.utils.wav import save_audio_to_file
from src.common.factory import EngineFactory, EngineClass
from src.common.logger import Logger
from src.common.session import Session
from src.common.types import SessionCtx, MODELS_DIR, RECORDS_DIR
from src.modules.speech.tts.chat_tts import ChatTTS

r"""
python -m unittest test.modules.speech.tts.test_chat.TestChatTTS.test_synthesize
python -m unittest test.modules.speech.tts.test_chat.TestChatTTS.test_synthesize_speak
STREAM=1 python -m unittest test.modules.speech.tts.test_chat.TestChatTTS.test_synthesize_speak
"""

r"""
<speak>
    <p>欢迎使用a chat bot!</p>
    <break time="1s"/>
    <p>SSML语音合成标记语言,待测试。</p>
    <emphasis level="strong">哇塞!</emphasis>
</speak>
"""


class TestChatTTS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tts_tag = os.getenv("LLM_TAG", "tts_chat")
        cls.tts_text = os.getenv("TTS_TEXT", "你好，我是机器人")
        cls.stream = os.getenv("STREAM", "")
        cls.compile = os.getenv("COMPILE", "")
        cls.source = os.getenv("SOURCE", "custom")
        cls.local_path = os.getenv("LOCAL_PATH", os.path.join(MODELS_DIR, "2Noise/ChatTTS"))
        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        kwargs = {}
        kwargs["local_path"] = self.local_path
        kwargs["source"] = self.source
        kwargs["compile"] = bool(self.compile)
        self.tts: ChatTTS = EngineFactory.get_engine_by_tag(EngineClass, self.tts_tag, **kwargs)
        self.session = Session(**SessionCtx("test_tts_client_id").__dict__)

    def tearDown(self):
        pass

    def test_synthesize(self):
        stream_info = self.tts.get_stream_info()
        print(stream_info)
        self.session.ctx.state["tts_text"] = self.tts_text
        print(self.session.ctx)
        iter = self.tts.synthesize_sync(self.session)
        res = bytearray()
        for i, chunk in enumerate(iter):
            print(i, len(chunk))
            res.extend(chunk)
        path = asyncio.run(
            save_audio_to_file(
                res,
                f"test_{self.tts.TAG}.wav",
                sample_rate=stream_info["rate"],
                sample_width=stream_info["sample_width"],
                channles=stream_info["channels"],
            )
        )
        print(path)

    def test_synthesize_speak(self):
        import pyaudio

        info = self.tts.get_stream_info()
        self.pyaudio_instance = pyaudio.PyAudio()
        self.audio_stream = self.pyaudio_instance.open(
            format=info["format"],
            channels=info["channels"],
            rate=info["rate"],
            output_device_index=None,
            output=True,
        )

        self.session.ctx.state["tts_text"] = self.tts_text
        print(self.session.ctx)
        self.tts.args.tts_stream = bool(self.stream)
        print(self.tts.args)
        iter = self.tts.synthesize_sync(self.session)
        sub_chunk_size = 1024
        for i, chunk in enumerate(iter):
            print(f"get {i} chunk {len(chunk)}")
            self.assertGreater(len(chunk), 0)
            if len(chunk) / sub_chunk_size < 100:
                self.audio_stream.write(chunk)
                continue
            for i in range(0, len(chunk), sub_chunk_size):
                sub_chunk = chunk[i : i + sub_chunk_size]
                self.audio_stream.write(sub_chunk)

        self.audio_stream.stop_stream()
        self.audio_stream.close()

        self.pyaudio_instance.terminate()
