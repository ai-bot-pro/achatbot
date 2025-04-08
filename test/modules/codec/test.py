import os
import logging

import unittest
import librosa
import soundfile
import torch
import torchaudio

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
SAMPLE_RATE=24000 CODEC_TAG=codec_transformers_mimi CODEC_MODEL_DIR=./models/kyutai/mimi \
    python -m unittest test.modules.codec.test.TestCodec.test_encode_decode
CODEC_TAG=codec_transformers_dac CODEC_MODEL_DIR=./models/descript/dac_16khz \
    python -m unittest test.modules.codec.test.TestCodec.test_encode_decode
SAMPLE_RATE=44100 CODEC_TAG=codec_transformers_dac CODEC_MODEL_DIR=./models/descript/dac_44khz \
    python -m unittest test.modules.codec.test.TestCodec.test_encode_decode
CODEC_TAG=codec_bitokenizer CODEC_MODEL_DIR=./models/SparkAudio/Spark-TTS-0.5B \
    python -m unittest test.modules.codec.test.TestCodec.test_encode_decode
SAMPLE_RATE=24000 CODEC_TAG=codec_snac CODEC_MODEL_DIR=./models/hubertsiuzdak/snac_24khz \
    python -m unittest test.modules.codec.test.TestCodec.test_encode_decode
SAMPLE_RATE=32000 CODEC_TAG=codec_snac CODEC_MODEL_DIR=./models/hubertsiuzdak/snac_32khz \
    python -m unittest test.modules.codec.test.TestCodec.test_encode_decode
SAMPLE_RATE=44100 CODEC_TAG=codec_snac CODEC_MODEL_DIR=./models/hubertsiuzdak/snac_44khz \
    python -m unittest test.modules.codec.test.TestCodec.test_encode_decode


# WavTokenizer small for speech 40 tokens/s
SAMPLE_RATE=24000 CODEC_TAG=codec_wavtokenizer \
    CODEC_MODEL_PATH=./models/novateur/WavTokenizer/WavTokenizer_small_600_24k_4096.ckpt \
    CODEC_CONFIG_PATH=./models/novateur/WavTokenizer/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml \
    python -m unittest test.modules.codec.test.TestCodec.test_encode_decode

# WavTokenizer small for speech 75 tokens/s
SAMPLE_RATE=24000 CODEC_TAG=codec_wavtokenizer \
    CODEC_MODEL_PATH=./models/novateur/WavTokenizer/WavTokenizer_small_320_24k_4096.ckpt \
    CODEC_CONFIG_PATH=./models/novateur/WavTokenizer/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml\
    python -m unittest test.modules.codec.test.TestCodec.test_encode_decode

# WavTokenizer medium for speech 75 tokens/s
SAMPLE_RATE=24000 CODEC_TAG=codec_wavtokenizer \
    CODEC_MODEL_PATH=./models/novateur/WavTokenizer-medium-speech-75token/wavtokenizer_medium_speech_320_24k_v2.ckpt \
            CODEC_CONFIG_PATH=./models/novateur/WavTokenizer-medium-speech-75token/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml \
    python -m unittest test.modules.codec.test.TestCodec.test_encode_decode

# WavTokenizer large for speech 40 tokens/s
SAMPLE_RATE=24000 CODEC_TAG=codec_wavtokenizer \
    CODEC_MODEL_PATH=./models/novateur/WavTokenizer-large-unify-40token/wavtokenizer_large_unify_600_24k.ckpt \
    CODEC_CONFIG_PATH=./models/novateur/WavTokenizer/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml \
    python -m unittest test.modules.codec.test.TestCodec.test_encode_decode

# ⭐️ WavTokenizer large for speech 75 tokens/s
SAMPLE_RATE=24000 CODEC_TAG=codec_wavtokenizer \
    CODEC_MODEL_PATH=./models/novateur/WavTokenizer-large-speech-75token/wavtokenizer_large_speech_320_v2.ckpt \
    CODEC_CONFIG_PATH=./models/novateur/WavTokenizer-medium-speech-75token/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml \
    python -m unittest test.modules.codec.test.TestCodec.test_encode_decode
"""


class TestCodec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # wget
        # https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav
        # -O records/asr_example_zh.wav
        audio_file = os.path.join(TEST_DIR, "audio_files/asr_example_zh.wav")
        cls.sample_rate = int(os.getenv("SAMPLE_RATE", "16000"))
        cls.audio_file = os.getenv("AUDIO_FILE", audio_file)
        cls.codec_tag = os.getenv("CODEC_TAG", "codec_xcodec2")
        Logger.init(os.getenv("LOG_LEVEL", "debug").upper(), is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        model_dir = os.path.join(MODELS_DIR, "HKUSTAudio/xcodec2")
        kwargs = CodecArgs(
            model_dir=os.getenv("CODEC_MODEL_DIR", model_dir),
            model_path=os.getenv("CODEC_MODEL_PATH", ""),
            config_path=os.getenv("CODEC_CONFIG_PATH", ""),
        ).__dict__
        self.codec: ICodec = CodecEnvInit.initCodecEngine(self.codec_tag, **kwargs)
        self.session = Session(**SessionCtx("test_codec_client_id").__dict__)

        pass

    def test_encode_decode(self):
        wav, sr = soundfile.read(self.audio_file)
        if sr != self.sample_rate:
            # Resample the sound file
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)
            print(f"resample rate: {sr} -> target rate: {self.sample_rate}")

        wav_tensor = torch.from_numpy(wav).float()  # Shape: (T)
        print(f"encode to vq codes from wav_tensor: {wav_tensor.shape}, rate: {self.sample_rate}")
        vq_code = self.codec.encode_code(wav_tensor)
        if isinstance(vq_code, list):
            for item in vq_code:
                print(f"vq_code: {item.shape}")
        else:
            print(f"vq_code: {vq_code.shape}")
        wav_tensor = self.codec.decode_code(vq_code)  # Shape: (T)
        print(f"decode vq_code to wav_tensor: {wav_tensor.shape}")

        wav_np = wav_tensor.detach().cpu().numpy()
        output_path = f"{self.codec_tag}_test_codec.wav"
        soundfile.write(output_path, wav_np, self.sample_rate)
        print(f"save to {output_path}")
