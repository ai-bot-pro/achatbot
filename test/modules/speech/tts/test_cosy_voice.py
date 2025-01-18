import os
import logging
import asyncio

import unittest

from src.modules.speech.tts.cosy_voice2_tts import CosyVoiceTTS
from src.common.factory import EngineFactory, EngineClass
from src.common.logger import Logger
from src.common.session import Session
from src.common.utils.wav import save_audio_to_file
from src.common.types import SessionCtx, CosyVoiceTTSArgs, MODELS_DIR

r"""
python -m unittest test.modules.speech.tts.test_cosy_voice.TestCosyVoiceTTS.test_get_voices

# cosyvoice default use sft infer
LOG_LEVEL=INFO TTS_TAG=tts_cosy_voice MODELS_DIR=/content/achatbot/models/FunAudioLLM/CosyVoice-300M-SFT \
  python -m unittest test.modules.speech.tts.test_cosy_voice.TestCosyVoiceTTS.test_synthesize

# cosyvoice test_synthesize with instruct infer
LOG_LEVEL=INFO TTS_TAG=tts_cosy_voice MODELS_DIR=/content/achatbot/models/FunAudioLLM/CosyVoice-300M-Instruct \
  TTS_TEXT="在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。" \
  INSTRUCT_TEXT="Theo \'Crimson\', is a fiery, passionate rebel leader. Fights with fervor for justice, but struggles with impulsiveness." \
  python -m unittest test.modules.speech.tts.test_cosy_voice.TestCosyVoiceTTS.test_synthesize

# cosyvoice test_synthesize with zero shot infer
# zero_shot usage, <|zh|><|en|><|jp|><|yue|><|ko|> for Chinese/English/Japanese/Cantonese/Korean
LOG_LEVEL=INFO TTS_TAG=tts_cosy_voice MODELS_DIR=/content/achatbot/models/FunAudioLLM/CosyVoice-300M \
  TTS_TEXT="您好啊，我是🤖机器人，很高兴认识您~" \
  REFERENCE_AUDIO_PATH=/content/achatbot/test/audio_files/asr_example_zh.wav \
  REFERENCE_TEXT="欢迎大家来体验达摩院推出的语音识别模型。" \
  python -m unittest test.modules.speech.tts.test_cosy_voice.TestCosyVoiceTTS.test_synthesize

# cosyvoice test_synthesize with cross_lingual infer
LOG_LEVEL=INFO TTS_TAG=tts_cosy_voice MODELS_DIR=/content/achatbot/models/FunAudioLLM/CosyVoice-300M \
  TTS_TEXT="<|en|>And then later on, fully acquiring that company. So keeping management in line, interest in line with the asset that\'s coming into the family is a reason why sometimes we don\'t buy the whole thing." \
  REFERENCE_AUDIO_PATH=/content/achatbot/test/audio_files/asr_example_zh.wav \
  python -m unittest test.modules.speech.tts.test_cosy_voice.TestCosyVoiceTTS.test_synthesize

# cosyvoice test_synthesize with voice convert infer
LOG_LEVEL=INFO TTS_TAG=tts_cosy_voice MODELS_DIR=/content/achatbot/models/FunAudioLLM/CosyVoice-300M \
  SRC_AUDIO_PATH=/content/achatbot/deps/CosyVoice/asset/cross_lingual_prompt.wav \
  REFERENCE_AUDIO_PATH=/content/achatbot/test/audio_files/asr_example_zh.wav \
  python -m unittest test.modules.speech.tts.test_cosy_voice.TestCosyVoiceTTS.test_synthesize


------

# cosyvoice2 no spk, so need reference audio
LOG_LEVEL=INFO TTS_TAG=tts_cosy_voice2 MODELS_DIR=/content/achatbot/models/FunAudioLLM/CosyVoice2-0.5B \
  REFERENCE_AUDIO_PATH=/content/achatbot/records/test_tts_cosy_voice.wav \
  python -m unittest test.modules.speech.tts.test_cosy_voice.TestCosyVoiceTTS.test_get_voices

# cosyvoice2 no spk, so need reference audio
LOG_LEVEL=INFO TTS_TAG=tts_cosy_voice2 MODELS_DIR=/content/achatbot/models/FunAudioLLM/CosyVoice2-0.5B \
  REFERENCE_AUDIO_PATH=/content/achatbot/records/test_tts_cosy_voice.wav \
  python -m unittest test.modules.speech.tts.test_cosy_voice.TestCosyVoiceTTS.test_get_voices

# cosyvoice2 test_synthesize with instruct2 infer
LOG_LEVEL=INFO TTS_TAG=tts_cosy_voice2 MODELS_DIR=/content/achatbot/models/FunAudioLLM/CosyVoice2-0.5B \
  REFERENCE_AUDIO_PATH=/content/achatbot/records/test_tts_cosy_voice.wav \
  INSTRUCT_TEXT="用四川话说这句话" \
  python -m unittest test.modules.speech.tts.test_cosy_voice.TestCosyVoiceTTS.test_synthesize

# cosyvoice2 test_synthesize with zero shot infer
LOG_LEVEL=INFO TTS_TAG=tts_cosy_voice2 MODELS_DIR=/content/achatbot/models/FunAudioLLM/CosyVoice2-0.5B \
  TTS_TEXT="您好啊，我是🤖机器人，很高兴认识您~" \
  REFERENCE_AUDIO_PATH=/content/achatbot/records/test_tts_cosy_voice.wav \
  REFERENCE_TEXT="你好，我是机器人, hello, test.modules.speech.tts.test_gtts.TestGTTS.test_synthesize" \
  python -m unittest test.modules.speech.tts.test_cosy_voice.TestCosyVoiceTTS.test_synthesize

# cosyvoice2 test_synthesize with cross lingual infer and use fine grained control, 
# for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
LOG_LEVEL=DEBUG TTS_TAG=tts_cosy_voice2 MODELS_DIR=/content/achatbot/models/FunAudioLLM/CosyVoice2-0.5B \
  TTS_TEXT="在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。" \
  REFERENCE_AUDIO_PATH=/content/achatbot/records/test_tts_cosy_voice.wav \
  python -m unittest test.modules.speech.tts.test_cosy_voice.TestCosyVoiceTTS.test_synthesize
"""


class TestCosyVoiceTTS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tts_tag = os.getenv("TTS_TAG", "tts_cosy_voice")
        cls.tts_text = os.getenv(
            "TTS_TEXT",
            "你好，我是机器人, hello, test.modules.speech.tts.test_gtts.TestGTTS.test_synthesize",
        )
        model_dir = os.path.join(MODELS_DIR, "CosyVoice-300M-SFT")
        cls.model_dir = os.getenv("MODELS_DIR", model_dir)
        Logger.init(os.getenv("LOG_LEVEL", "debug").upper(), is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        kwargs = CosyVoiceTTSArgs(
            model_dir=self.model_dir,
            reference_text=os.getenv("REFERENCE_TEXT", ""),
            reference_audio_path=os.getenv("REFERENCE_AUDIO_PATH", ""),
            src_audio_path=os.getenv("SRC_AUDIO_PATH", ""),
            instruct_text=os.getenv("INSTRUCT_TEXT", ""),
            spk_id=os.getenv(
                "SPK_ID", ""
            ),  # for cosyvoice sft/struct inference, cosyvoice2 don't use it
        ).__dict__
        self.tts: CosyVoiceTTS = EngineFactory.get_engine_by_tag(
            EngineClass, self.tts_tag, **kwargs
        )
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
