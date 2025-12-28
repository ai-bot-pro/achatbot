import os

import numpy as np
import soundfile
import unittest

from src.common.interface import ITts
from src.modules.speech.tts import TTSEnvInit
from src.common.logger import Logger
from src.common.session import Session
from src.common.types import RECORDS_DIR, SessionCtx, TEST_DIR

r"""
python -m unittest test.modules.speech.tts.test_glm.TestGLMTTS.test_get_voices
python -m unittest test.modules.speech.tts.test_glm.TestGLMTTS.test_set_voice
python -m unittest test.modules.speech.tts.test_glm.TestGLMTTS.test_synthesize
python -m unittest test.modules.speech.tts.test_glm.TestGLMTTS.test_synthesize_speak


# voice
# https://github.com/weedge/GLM-TTS/blob/main/examples/example_zh.jsonl
TTS_GLM_VOICE=jiayan_zh \
    TTS_GLM_PROMPT_SPEECH_PATH=https://raw.githubusercontent.com/weedge/GLM-TTS/refs/heads/main/examples/prompt/jiayan_zh.wav \
    TTS_GLM_PROMPT_TEXT="他当时还跟线下其他的站姐吵架，然后，打架进局子了。" \
    python -m unittest test.modules.speech.tts.test_glm.TestGLMTTS.test_synthesize

# https://github.com/weedge/GLM-TTS/blob/main/examples/example_en.jsonl
TTS_GLM_VOICE=jiayan_en \
    TTS_GLM_PROMPT_SPEECH_PATH=https://raw.githubusercontent.com/weedge/GLM-TTS/refs/heads/main/examples/prompt/jiayan_en.wav \
    TTS_GLM_PROMPT_TEXT="I wonder if you'd like to have a burger with me." \
    python -m unittest test.modules.speech.tts.test_glm.TestGLMTTS.test_synthesize
"""


class TestGLMTTS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tts_tag = os.getenv("TTS_GLM_TAG", "tts_glm")
        cls.tts_voice = os.getenv("TTS_GLM_VOICE", "jiayan_zh")
        cls.tts_prompt_text = os.getenv(
            "TTS_GLM_PROMPT_TEXT", "他当时还跟线下其他的站姐吵架，然后，打架进局子了。"
        )
        cls.tts_prompt_speech_path = os.getenv(
            "TTS_GLM_PROMPT_SPEECH_PATH",
            "https://raw.githubusercontent.com/weedge/GLM-TTS/refs/heads/main/examples/prompt/jiayan_zh.wav",
        )
        cls.tts_text = os.getenv(
            "TTS_TEXT",
            # "你好，hello.",
            "GLM-TTS is a high-quality text-to-speech (TTS) synthesis system based on large language models, supporting zero-shot voice cloning and streaming inference. This system adopts a two-stage architecture: first, it uses LLM to generate speech token sequences, then uses Flow model to convert tokens into high-quality audio waveforms. By introducing a Multi-Reward Reinforcement Learning framework, GLM-TTS can generate more expressive and emotional speech, significantly improving the expressiveness of traditional TTS systems.",
        )

        Logger.init(os.getenv("LOG_LEVEL", "debug").upper(), is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.tts: ITts = TTSEnvInit.initTTSEngine(self.tts_tag)
        self.session = Session(**SessionCtx("test_tts_client_id").__dict__)
        self.pyaudio_instance = None
        self.audio_stream = None

    def tearDown(self):
        self.audio_stream and self.audio_stream.stop_stream()
        self.audio_stream and self.audio_stream.close()
        self.pyaudio_instance and self.pyaudio_instance.terminate()

    def test_get_voices(self):
        voices = self.tts.get_voices()
        self.assertGreaterEqual(len(voices), 0)
        print(voices)

    def test_set_voice(self):
        self.tts.set_voice(
            voice_name=self.tts_voice,
            prompt_text=self.tts_prompt_text,
            prompt_speech_path=self.tts_prompt_speech_path,
        )
        voices = self.tts.get_voices()
        self.assertGreaterEqual(len(voices), 0)
        print(voices)

    def test_synthesize(self):
        self.session.ctx.state["tts_text"] = self.tts_text
        print(self.session.ctx)
        iter = self.tts.synthesize_sync(self.session)
        res = bytearray()
        for i, chunk in enumerate(iter):
            print(i, len(chunk))
            res.extend(chunk)

        stream_info = self.tts.get_stream_info()
        print(f"stream_info:{stream_info}")

        file_name = f"test_{self.tts.TAG}.wav"
        os.makedirs(RECORDS_DIR, exist_ok=True)
        file_path = os.path.join(RECORDS_DIR, file_name)
        data = np.frombuffer(res, dtype=stream_info["np_dtype"])
        soundfile.write(file_path, data, stream_info["rate"])

        print(file_path)

    def test_synthesize_speak(self):
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
