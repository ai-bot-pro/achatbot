from io import BytesIO
import os
import logging

import unittest
import librosa
import numpy as np
import soundfile as sf

from src.common.utils.audio_utils import bytes2NpArrayWith16
from src.common.utils.wav import read_wav_to_bytes
from src.common.logger import Logger
from src.common.types import TEST_DIR

from dotenv import load_dotenv


load_dotenv(override=True)

"""
python -m unittest test.common.test_bytes2np.TestBytes2Numpy.test_bytes2np

"""


def geneHeadInfo(sampleRate, bits, sampleNum):
    import struct

    rHeadInfo = b"\x52\x49\x46\x46"
    fileLength = struct.pack("i", sampleNum + 36)
    rHeadInfo += fileLength
    rHeadInfo += b"\x57\x41\x56\x45\x66\x6d\x74\x20\x10\x00\x00\x00\x01\x00\x01\x00"
    rHeadInfo += struct.pack("i", sampleRate)
    rHeadInfo += struct.pack("i", int(sampleRate * bits / 8))
    rHeadInfo += b"\x02\x00"
    rHeadInfo += struct.pack("H", bits)
    rHeadInfo += b"\x64\x61\x74\x61"
    rHeadInfo += struct.pack("i", sampleNum)
    return rHeadInfo


class TestBytes2Numpy(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # use for gpu to test
        # assert torch.cuda.is_available()

        cls._debug = os.getenv("DEBUG", "false").lower() == "true"
        cls.force_preparation = os.getenv("FORCE_PREPARATION", "false").lower() == "true"

        # https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/asr/test_audio/asr_example_zh.wav
        audio_file = os.path.join(TEST_DIR, "audio_files/asr_example_zh.wav")
        cls.audio_file = os.getenv("AUDIO_FILE", audio_file)
        cls.data_bytes, cls.sr = read_wav_to_bytes(cls.audio_file)

        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    async def asyncSetUp(self):
        pass

    async def asyncTearDown(self):
        pass

    async def test_bytes2np(self):
        audio_np_data = bytes2NpArrayWith16(self.data_bytes)
        print(f"{audio_np_data=}")

        headinfo = geneHeadInfo(16000, 16, len(self.data_bytes))
        input_audio_byte = headinfo + self.data_bytes
        input_audio, sr = sf.read(BytesIO(input_audio_byte), dtype="float32")
        print(f"{input_audio=} {sr=}")

        librosa_output, sampling_rate = librosa.load(self.audio_file, sr=16000)
        print(f"{librosa_output=}, {sampling_rate=}")

        assert (input_audio == librosa_output).all()
        assert (input_audio == audio_np_data).all()
