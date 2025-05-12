import io
import os
import logging
import asyncio

import numpy as np
import soundfile
import unittest

from src.common.interface import ITts
from src.modules.speech.tts import TTSEnvInit
from src.common.factory import EngineFactory, EngineClass
from src.common.logger import Logger
from src.common.session import Session
from src.common.utils.wav import save_audio_to_file
from src.common.types import RECORDS_DIR, SessionCtx, MODELS_DIR

r"""
LLM_MODEL_NAME_OR_PATH=./models/VITA-MLLM/VITA-Audio-Plus-Vanilla \
    AUDIO_TOKENIZER_TYPE=sensevoice_glm4voice \
    FLOW_PATH=./models/THUDM/glm-4-voice-decoder \
    LLM_DEVICE=cuda LLM_TORCH_DTYPE=bfloat16 LLM_ATTN_IMP=flash_attention_2 \
    python -m unittest test.modules.speech.tts.test_vita.TestVITATTS.test_synthesize
"""


class TestVITATTS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tts_tag = os.getenv("TTS_TAG", "tts_vita")
        cls.tts_text = os.getenv(
            "TTS_TEXT",
            "你好，hello.",
        )

        Logger.init(os.getenv("LOG_LEVEL", "debug").upper(), is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        kwargs = {}
        self.tts: EngineClass | ITts = TTSEnvInit.initTTSEngine(self.tts_tag, **kwargs)
        self.session = Session(**SessionCtx("test_tts_client_id").__dict__)
        self.pyaudio_instance = None
        self.audio_stream = None

    def tearDown(self):
        self.audio_stream and self.audio_stream.stop_stream()
        self.audio_stream and self.audio_stream.close()
        self.pyaudio_instance and self.pyaudio_instance.terminate()

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
