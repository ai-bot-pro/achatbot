import os
import logging

import unittest
import soundfile

from src.common.logger import Logger
from src.common.session import Session
from src.common.types import MODELS_DIR, SessionCtx
from src.types.codec import CodecArgs
from src.modules.codec import CodecEnvInit, ICodec

r"""
CODEC_TAG=codec_xcodec2 python -m unittest test.modules.codec.test.TestCodec.test_encode_decode
CODEC_TAG=codec_moshi_mimi python -m unittest test.modules.codec.test.TestCodec.test_encode_decode
CODEC_TAG=codec_transformers_mimi python -m unittest test.modules.codec.test.TestCodec.test_encode_decode
"""


class TestCodec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.codec_tag = os.getenv("CODEC_TAG", "codec_xcodec2")
        Logger.init(os.getenv("LOG_LEVEL", "debug").upper(), is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        model_dir = os.path.join(MODELS_DIR, "HKUSTAudio/xcodec2")
        kwargs = CodecArgs(model_dir=os.getenv("CODEC_MODEL_DIR", model_dir)).__dict__
        self.codec: ICodec = CodecEnvInit.initCodecEngine(self.codec_tag, **kwargs)
        self.session = Session(**SessionCtx("test_codec_client_id").__dict__)

    def tearDown(self):
        pass

    def test_encode_decode(self):
        vq_code = self.codec.encode_code(self.session)
        wav_tensor, _ = self.codec.decode_code(vq_code)

        wav_np = wav_tensor.detach().cpu().numpy()
        soundfile.write("test_codec.wav", wav_np, 24000)
