import io
import os
import logging
import asyncio

import numpy as np
import soundfile
import unittest

from src.modules.speech.tts.f5_tts import F5TTS
from src.common.factory import EngineFactory, EngineClass
from src.common.logger import Logger
from src.common.session import Session
from src.common.utils.wav import save_audio_to_file
from src.common.types import RECORDS_DIR, SessionCtx, MODELS_DIR
from src.types.speech.tts.f5 import F5TTSArgs, F5TTSDiTModelConfig

r"""
python -m unittest test.modules.speech.tts.test_f5.TestF5TTS.test_get_voices
python -m unittest test.modules.speech.tts.test_f5.TestF5TTS.test_synthesize
python -m unittest test.modules.speech.tts.test_f5.TestF5TTS.test_synthesize_speak
"""


class TestF5TTS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tts_tag = os.getenv("TTS_TAG", "tts_f5")
        cls.tts_text = os.getenv(
            "TTS_TEXT",
            "你好，hello.",
        )

        cls.voc_name = os.getenv("VOC_NAME", "vocos")
        cls.model_type = os.getenv("MODEL_TYPE", "F5-TTS")
        vocos_model_dir = os.path.join(MODELS_DIR, "charactr/vocos-mel-24khz")
        cls.vocoder_model_dir = os.getenv("VOCODER_MODEL_DIR", vocos_model_dir)
        tts_model_ckpt_path = os.path.join(
            MODELS_DIR, "SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"
        )
        if cls.voc_name == "bigvgan":
            tts_model_ckpt_path = os.path.join(
                MODELS_DIR, "SWivid/F5-TTS/F5TTS_Base_bigvgan/model_1200000.safetensors"
            )
        if cls.model_type == "E2-TTS":
            if cls.voc_name == "vocos":
                tts_model_ckpt_path = os.path.join(
                    MODELS_DIR, "SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"
                )
            else:
                raise Exception("E2-TTS just support vocos vocoder")
        cls.model_ckpt_path = os.getenv("MODEL_CKPT_PATH", tts_model_ckpt_path)

        Logger.init(os.getenv("LOG_LEVEL", "debug").upper(), is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        kwargs = F5TTSArgs(
            model_type=self.model_type,
            model_cfg=F5TTSDiTModelConfig().__dict__,
            model_ckpt_path=self.model_ckpt_path,
            vocab_file=os.getenv("VOCAB_FILE", ""),
            vocoder_name=self.voc_name,
            vocoder_ckpt_dir=self.vocoder_model_dir,
            ref_audio_file=os.getenv("REFERENCE_AUDIO_PATH", ""),
            ref_text=os.getenv("REFERENCE_TEXT", ""),
        ).__dict__
        self.tts: F5TTS = EngineFactory.get_engine_by_tag(EngineClass, self.tts_tag, **kwargs)
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
        self.session.ctx.state["tts_text"] = self.tts_text
        print(self.session.ctx)
        iter = self.tts.synthesize_sync(self.session)
        res = bytearray()
        for i, chunk in enumerate(iter):
            print(i, len(chunk))
            res.extend(chunk)

        stream_info = self.tts.get_stream_info()
        print(f"stream_info:{stream_info}")

        ## for np.int16
        # file_path = asyncio.run(
        #    save_audio_to_file(
        #        res,
        #        f"test_{self.tts.TAG}.wav",
        #        sample_rate=stream_info["rate"],
        #        sample_width=stream_info["sample_width"],
        #        channles=stream_info["channels"],
        #    )
        # )

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
