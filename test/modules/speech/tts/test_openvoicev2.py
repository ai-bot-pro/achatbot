import io
import os
import logging
import asyncio

import numpy as np
import soundfile
import unittest

from src.common.factory import EngineFactory, EngineClass
from src.common.logger import Logger
from src.common.session import Session
from src.common.utils.wav import save_audio_to_file
from src.common.types import ASSETS_DIR, RECORDS_DIR, SessionCtx, MODELS_DIR
from src.types.speech.tts.openvoicev2 import OpenVoiceV2TTSArgs
from src.modules.speech.tts.openvoicev2_tts import OpenVoiceV2TTS

r"""
python -m unittest test.modules.speech.tts.test_openvoicev2.TestOpenVoiceV2TTS.test_set_voice
python -m unittest test.modules.speech.tts.test_openvoicev2.TestOpenVoiceV2TTS.test_get_voices
python -m unittest test.modules.speech.tts.test_openvoicev2.TestOpenVoiceV2TTS.test_synthesize
python -m unittest test.modules.speech.tts.test_openvoicev2.TestOpenVoiceV2TTS.test_clone_synthesize
python -m unittest test.modules.speech.tts.test_openvoicev2.TestOpenVoiceV2TTS.test_synthesize_speak
python -m unittest test.modules.speech.tts.test_openvoicev2.TestOpenVoiceV2TTS.test_clone_synthesize_speak
"""


class TestOpenVoiceV2TTS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tts_tag = os.getenv("TTS_TAG", "tts_openvoicev2")
        cls.tts_text = os.getenv(
            "TTS_TEXT",
            "hello,你好！我就是那个万人敬仰的太乙真人。",
        )
        Logger.init(os.getenv("LOG_LEVEL", "debug").upper(), is_file=False)

        # cls.language = os.getenv("LANG", "ZH") # LANG sys env param, don't use it
        cls.language = os.getenv("LANGUAGE", "ZH")

        # download melo-tts model ckpt
        # huggingface-cli download myshell-ai/MeloTTS-English-v3 --local-dir ./models/myshell-ai/MeloTTS-English-v3
        # huggingface-cli download myshell-ai/MeloTTS-Chinese --local-dir ./models/myshell-ai/MeloTTS-Chinese
        tts_config_path = os.path.join(MODELS_DIR, "myshell-ai/MeloTTS-Chinese/config.json")
        tts_ckpt_path = os.path.join(MODELS_DIR, "myshell-ai/MeloTTS-Chinese/checkpoint.pth")
        if cls.language == "EN_NEWEST":
            tts_config_path = os.path.join(MODELS_DIR, "myshell-ai/MeloTTS-English-v3/config.json")
            tts_ckpt_path = os.path.join(MODELS_DIR, "myshell-ai/MeloTTS-English-v3/checkpoint.pth")

        cls.tts_ckpt_path = os.getenv("TTS_CKPT_PATH", tts_ckpt_path)
        cls.tts_conf_path = os.getenv("TTS_CONF_PATH", tts_config_path)

        # download openvoice converter model ckpt
        # huggingface-cli download myshell-ai/OpenVoiceV2 --local-dir ./models/myshell-ai/OpenVoiceV2
        converter_conf_path = os.path.join(
            MODELS_DIR, "myshell-ai/OpenVoiceV2/converter/config.json"
        )
        converter_ckpt_path = os.path.join(
            MODELS_DIR, "myshell-ai/OpenVoiceV2/converter/checkpoint.pth"
        )
        cls.converter_ckpt_path = os.getenv("CONVERTER_CKPT_PATH", converter_ckpt_path)
        cls.converter_conf_path = os.getenv("CONVERTER_CONF_PATH", converter_conf_path)

        # the tone color converter stats
        speaker_key = cls.language.lower().replace("_", "-")
        src_se_ckpt_path = os.path.join(
            MODELS_DIR, f"myshell-ai/OpenVoiceV2/base_speakers/ses/{speaker_key}.pth"
        )
        cls.src_se_ckpt_path = os.getenv("SRC_SE_CKPT_PATH", src_se_ckpt_path)

        # for extra target tone color
        cls.target_se_ckpt_path = os.getenv("TARGET_SE_CKPT_PATH", "")
        cls.target_audio_path = os.getenv(
            "TARGET_AUDIO_PATH", os.path.join(ASSETS_DIR, "basic_ref_zh.wav")
        )

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        kwargs = OpenVoiceV2TTSArgs(
            language=self.language,
            tts_ckpt_path=self.tts_ckpt_path,
            tts_config_path=self.tts_conf_path,
            enable_clone=bool(os.getenv("ENABLE_CLONE", "")),
            converter_ckpt_path=self.converter_ckpt_path,
            converter_conf_path=self.converter_conf_path,
            src_se_ckpt_path=self.src_se_ckpt_path,
            is_save=bool(os.getenv("IS_SAVE", "")),
        ).__dict__
        self.tts: OpenVoiceV2TTS = EngineFactory.get_engine_by_tag(
            EngineClass, self.tts_tag, **kwargs
        )
        self.session = Session(**SessionCtx("test_tts_client_id").__dict__)
        self.pyaudio_instance = None
        self.audio_stream = None

    def tearDown(self):
        self.audio_stream and self.audio_stream.stop_stream()
        self.audio_stream and self.audio_stream.close()
        self.pyaudio_instance and self.pyaudio_instance.terminate()

    def test_set_voice(self):
        self.tts.set_voice(self.target_audio_path)

        print(self.tts.tts_model.hps.data.spk2id)
        self.assertGreater(len(self.tts.tts_model.hps.data.spk2id), 1)

        self.assertEqual(hasattr(self.tts.tts_model.hps.data.spk2id, "custom"), True)
        print(self.tts.tts_model.hps.data.spk2id["custom"])
        self.assertGreater(len(self.tts.tts_model.hps.data.spk2id["custom"]), 0)

        self.assertIsNot(self.tts.target_se_stats_tensor, None)
        print(self.tts.target_se_stats_tensor.shape)
        self.assertGreater(self.tts.target_se_stats_tensor.shape[1], 1)

    def test_get_voices(self):
        voices = self.tts.get_voices()
        print("set_voice before:", voices)
        self.assertGreater(len(voices), 0)

        self.tts.set_voice(self.target_audio_path)

        voices = self.tts.get_voices()
        print("set_voice after:", voices)
        self.assertGreater(len(voices), 1)

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

        file_name = f"test_tts_melo.wav"
        os.makedirs(RECORDS_DIR, exist_ok=True)
        file_path = os.path.join(RECORDS_DIR, file_name)
        data = np.frombuffer(res, dtype=stream_info["np_dtype"])
        soundfile.write(file_path, data, stream_info["rate"])

        print(file_path)

    def test_clone_synthesize(self):
        self.session.ctx.state["tts_text"] = self.tts_text
        print(self.session.ctx)

        self.tts.args.enable_clone = True
        self.assertEqual(self.tts.args.enable_clone, True)

        # set clone target audio
        self.test_set_voice()
        self.assertIsNot(self.tts.src_se_stats_tensor, None)
        self.assertIsNot(self.tts.target_se_stats_tensor, None)

        iter = self.tts.synthesize_sync(self.session)
        res = bytearray()
        for i, chunk in enumerate(iter):
            print(i, len(chunk))
            res.extend(chunk)

        stream_info = self.tts.get_stream_info()
        print(f"stream_info:{stream_info}")

        file_name = f"test_openvoicev2_clone.wav"
        os.makedirs(RECORDS_DIR, exist_ok=True)
        file_path = os.path.join(RECORDS_DIR, file_name)
        data = np.frombuffer(res, dtype=stream_info["np_dtype"])
        soundfile.write(file_path, data, stream_info["rate"])

        print(file_path)

    def test_synthesize_speak(self):
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

    def test_clone_synthesize_speak(self):
        import pyaudio

        self.tts.args.enable_clone = True
        self.assertEqual(self.tts.args.enable_clone, True)

        # set clone target audio
        self.test_set_voice()
        self.assertIsNot(self.tts.src_se_stats_tensor, None)
        self.assertIsNot(self.tts.target_se_stats_tensor, None)

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
