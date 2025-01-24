import io
import os
import logging
import asyncio

import numpy as np
import soundfile
import unittest

from src.modules.speech.tts.fish_speech_tts import FishSpeechTTS
from src.common.factory import EngineFactory, EngineClass
from src.common.logger import Logger
from src.common.session import Session
from src.common.utils.wav import save_audio_to_file
from src.common.types import RECORDS_DIR, SessionCtx, MODELS_DIR, TEST_DIR
from src.types.speech.tts.fish_speech import FishSpeechTTSArgs

r"""
python -m unittest test.modules.speech.tts.test_fish_speech.TestFishSpeechTTS.test_get_voices
python -m unittest test.modules.speech.tts.test_fish_speech.TestFishSpeechTTS.test_set_voice
python -m unittest test.modules.speech.tts.test_fish_speech.TestFishSpeechTTS.test_synthesize
python -m unittest test.modules.speech.tts.test_fish_speech.TestFishSpeechTTS.test_synthesize_speak

# warm up
FS_WARM_UP_TEXT="hello weedge" python -m unittest test.modules.speech.tts.test_fish_speech.TestFishSpeechTTS.test_synthesize
FS_WARM_UP_TEXT="hello weedge" python -m unittest test.modules.speech.tts.test_fish_speech.TestFishSpeechTTS.test_synthesize_speak

# use ref audio
FS_REFERENCE_AUDIO_PATH=./test/audio_files/asr_example_zh.wav \
    TTS_TEXT="好的，我们从扩散过程开始。它是一个马尔可夫过程，其中数据 x 通过逐步添加噪声转化为 z。" \
    python -m unittest test.modules.speech.tts.test_fish_speech.TestFishSpeechTTS.test_synthesize
FS_REFERENCE_AUDIO_PATH=./test/audio_files/asr_example_zh.wav \
    TTS_TEXT="好的，我们从扩散过程开始。它是一个马尔可夫过程，其中数据 x 通过逐步添加噪声转化为 z。" \
    python -m unittest test.modules.speech.tts.test_fish_speech.TestFishSpeechTTS.test_synthesize_speak

# warm up; use ref audio and prompt text
FS_REFERENCE_AUDIO_PATH=./test/audio_files/asr_example_zh.wav  \
    FS_REFERENCE_TEXT="高兴的说出内容" \
    FS_WARM_UP_TEXT="hello weedge" \
    TTS_TEXT="好的，我们从扩散过程开始。它是一个马尔可夫过程，其中数据 x 通过逐步添加噪声转化为 z。 " \
    python -m unittest test.modules.speech.tts.test_fish_speech.TestFishSpeechTTS.test_synthesize

# warm up; use ref audio and prompt text; long text
FS_REFERENCE_AUDIO_PATH=./test/audio_files/asr_example_zh.wav \
    FS_REFERENCE_TEXT="高兴的说出内容" \
    FS_WARM_UP_TEXT="hello weedge" \
    TTS_TEXT="好的，我们从扩散过程开始。它是一个马尔可夫过程，其中数据 x 通过逐步添加噪声转化为 z。
这个过程可以用公式表示为 q(z|x) = N(αx, σ^2I)，其中 α 和 σ 是控制噪声量的参数。
反向过程是从噪声 z 恢复数据 x 的过程，通过神经网络估计得到 x_θ(z)。
训练目标是最小化噪声估计误差，也就是 ϵ_θ(z) 与实际噪声 ϵ 之间的差距。
关键点在于训练了条件模型和无条件模型，然后通过公式混合它们的 score。
公式可以简化为 ϵ_tilde = (1+w)ϵ_conditional - wϵ_unconditional, w 是引导的强度。 " \
    python -m unittest test.modules.speech.tts.test_fish_speech.TestFishSpeechTTS.test_synthesize

# warm up; use ref audio and prompt text; long text to speak
FS_REFERENCE_AUDIO_PATH=./test/audio_files/asr_example_zh.wav \
    FS_REFERENCE_TEXT="高兴的说出内容" \
    FS_WARM_UP_TEXT="hello weedge" \
    TTS_TEXT="好的，我们从扩散过程开始。它是一个马尔可夫过程，其中数据 x 通过逐步添加噪声转化为 z。
这个过程可以用公式表示为 q(z|x) = N(αx, σ^2I)，其中 α 和 σ 是控制噪声量的参数。
反向过程是从噪声 z 恢复数据 x 的过程，通过神经网络估计得到 x_θ(z)。
训练目标是最小化噪声估计误差，也就是 ϵ_θ(z) 与实际噪声 ϵ 之间的差距。
关键点在于训练了条件模型和无条件模型，然后通过公式混合它们的 score。
公式可以简化为 ϵ_tilde = (1+w)ϵ_conditional - wϵ_unconditional, w 是引导的强度。 " \
    python -m unittest test.modules.speech.tts.test_fish_speech.TestFishSpeechTTS.test_synthesize_speak

fish-speech 对公式的支持还不够好，需要微调，加入朗读公式的音频文本数据进行训练
"""


class TestFishSpeechTTS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tts_tag = os.getenv("TTS_TAG", "tts_fishspeech")
        cls.tts_text = os.getenv(
            "TTS_TEXT",
            "你好，hello.",
        )

        lm_checkpoint_dir = os.path.join(MODELS_DIR, "fishaudio/fish-speech-1.5")
        cls.lm_checkpoint_dir = os.getenv("FS_LM_CHECKPOINT_DIR", lm_checkpoint_dir)
        gan_checkpoint_path = os.path.join(
            MODELS_DIR,
            "fishaudio/fish-speech-1.5",
            "firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
        )
        cls.gan_checkpoint_path = os.getenv("FS_GAN_CHECKPOINT_PATH", gan_checkpoint_path)

        Logger.init(os.getenv("LOG_LEVEL", "debug").upper(), is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        kwargs = FishSpeechTTSArgs(
            warm_up_text=os.getenv("FS_WARM_UP_TEXT", ""),
            lm_checkpoint_dir=self.lm_checkpoint_dir,
            gan_checkpoint_path=self.gan_checkpoint_path,
            gan_config_path=os.getenv(
                "FS_GAN_CONFIG_PATH",
                "../../../../deps/FishSpeech/fish_speech/configs",
            ),
            ref_audio_path=os.getenv("FS_REFERENCE_AUDIO_PATH", None),
            ref_text=os.getenv("FS_REFERENCE_TEXT", ""),
        ).__dict__
        self.tts: FishSpeechTTS = EngineFactory.get_engine_by_tag(
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

    def test_set_voice(self):
        voices = self.tts.get_voices()
        self.assertGreaterEqual(len(voices), 0)
        print(voices)

        self.tts.set_voice(os.path.join(TEST_DIR, "audio_files/asr_example_zh.wav"))
        add_voices = self.tts.get_voices()
        self.assertEqual(len(add_voices), len(voices) + 1)
        print(add_voices)

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
