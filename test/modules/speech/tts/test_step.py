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
# ---- TTS_MODE: voice_clone ----

python -m unittest test.modules.speech.tts.test_step.TestStepTTS.test_get_voices
REF_AUDIO_PATH=./test/audio_files/asr_example_zh.wav \
    REF_TEXT="欢迎大家来体验达摩院推出的语音识别模型" \
    python -m unittest test.modules.speech.tts.test_step.TestStepTTS.test_set_voice

python -m unittest test.modules.speech.tts.test_step.TestStepTTS.test_synthesize
python -m unittest test.modules.speech.tts.test_step.TestStepTTS.test_synthesize_speak

# ref audio 
TTS_STREAM_FACTOR=4 \
    REF_AUDIO_PATH=./test/audio_files/asr_example_zh.wav \
    REF_TEXT="欢迎大家来体验达摩院推出的语音识别模型" \
    TTS_TEXT="万物之始,大道至简,衍化至繁。君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。" \
    python -m unittest test.modules.speech.tts.test_step.TestStepTTS.test_synthesize

TTS_STREAM_FACTOR=4 \
REF_AUDIO_PATH=./test/audio_files/asr_example_zh.wav \
    REF_TEXT="欢迎大家来体验达摩院推出的语音识别模型" \
    TTS_TEXT="万物之始,大道至简,衍化至繁。君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。" \
    python -m unittest test.modules.speech.tts.test_step.TestStepTTS.test_synthesize_speak

# ---- TTS_MODE: voice_clone ----
# src audio + default ref audio
SRC_AUDIO_PATH=./test/audio_files/asr_example_zh.wav \
    python -m unittest test.modules.speech.tts.test_step.TestStepTTS.test_synthesize

"""


class TestStepTTS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tts_tag = os.getenv("TTS_TAG", "tts_step")
        cls.src_audio_path = os.getenv(
            "SRC_AUDIO_PATH",
            # os.path.join(TEST_DIR, "audio_files/asr_example_zh.wav"),
            "",
        )
        cls.ref_audio_path = os.getenv(
            "REF_AUDIO_PATH",
            # os.path.join(TEST_DIR, "audio_files/asr_example_zh.wav"),
            "",
        )
        cls.ref_text = os.getenv(
            "REF_TEXT",
            # "欢迎大家来体验达摩院推出的语音识别模型",
            "",
        )
        cls.tts_text = os.getenv(
            "TTS_TEXT",
            "你好，hello.",
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
        voices = self.tts.get_voices()
        self.assertGreaterEqual(len(voices), 0)
        print(voices)

        self.tts.set_voice(
            self.ref_audio_path,
            ref_speaker="test_speaker",
            ref_text=self.ref_text,
        )
        add_voices = self.tts.get_voices()
        self.assertEqual(len(add_voices), len(voices) + 1)
        print(add_voices)

    def test_synthesize(self):
        ref_speaker = ""
        if os.path.exists(self.ref_audio_path):
            ref_speaker = "test_speaker"
            self.tts.set_voice(
                self.ref_audio_path,
                ref_speaker=ref_speaker,
                ref_text=self.ref_text,
            )
            self.session.ctx.state["ref_speaker"] = ref_speaker
        else:
            voices = self.tts.get_voices()
            self.assertGreaterEqual(len(voices), 0)
            print(f"use default voices: {voices}")

        if os.path.exists(self.src_audio_path):
            self.session.ctx.state["src_audio_path"] = self.src_audio_path

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

        ref_speaker = ""
        if os.path.exists(self.ref_audio_path):
            ref_speaker = "test_speaker"
            self.tts.set_voice(
                self.ref_audio_path,
                ref_speaker=ref_speaker,
                ref_text=self.ref_text,
            )
            self.session.ctx.state["ref_speaker"] = ref_speaker
        else:
            voices = self.tts.get_voices()
            self.assertGreaterEqual(len(voices), 0)
            print(f"use default voices: {voices}")

        if os.path.exists(self.src_audio_path):
            self.session.ctx.state["src_audio_path"] = self.src_audio_path

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
