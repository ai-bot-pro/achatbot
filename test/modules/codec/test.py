import os
import logging

import unittest
import soundfile
import torch

from src.common.logger import Logger
from src.common.session import Session
from src.common.types import MODELS_DIR, TEST_DIR, SessionCtx
from src.types.codec import CodecArgs
from src.modules.codec import CodecEnvInit, ICodec

r"""
CODEC_TAG=codec_xcodec2 CODEC_MODEL_DIR=./models/HKUSTAudio/xcodec2 \
    python -m unittest test.modules.codec.test.TestCodec.test_encode_decode
CODEC_TAG=codec_moshi_mimi CODEC_MODEL_DIR=./models/kyutai/moshiko-pytorch-bf16 \
    python -m unittest test.modules.codec.test.TestCodec.test_encode_decode
CODEC_TAG=codec_transformers_mimi CODEC_MODEL_DIR=./models/kyutai/mimi \
    python -m unittest test.modules.codec.test.TestCodec.test_encode_decode
"""


class TestCodec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # wget
        # https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav
        # -O records/asr_example_zh.wav
        audio_file = os.path.join(TEST_DIR, "audio_files/asr_example_zh.wav")
        cls.audio_file = os.getenv("AUDIO_FILE", audio_file)
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
        wav, sr = soundfile.read(self.audio_file)
        wav_tensor = torch.from_numpy(wav).float()  # Shape: (T)
        print(f"encode to vq codes from wav_tensor: {wav_tensor.shape}")
        vq_code = self.codec.encode_code(wav_tensor)
        print(f"vq_code: {vq_code.shape}")
        wav_tensor = self.codec.decode_code(vq_code)  # Shape: (T)
        print(f"decode vq_code to wav_tensor: {wav_tensor.shape}")

        wav_np = wav_tensor.detach().cpu().numpy()
        output_path = f"{self.codec_tag}_test_codec.wav"
        soundfile.write(output_path, wav_np, sr)
        print(f"save to {output_path}")
