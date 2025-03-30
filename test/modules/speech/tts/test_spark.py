import os
from time import perf_counter

import librosa
import numpy as np
import soundfile
import unittest

from src.common.interface import ITts
from src.modules.speech.tts import TTSEnvInit
from src.common.logger import Logger
from src.common.session import Session
from src.common.types import RECORDS_DIR, SessionCtx, TEST_DIR

"""
# download hf model and tokenizer vocab
huggingface-cli download SparkAudio/Spark-TTS-0.5B --quie --local-dir ./models/SparkAudio/Spark-TTS-0.5B

# download converated gguf (f16/Q8/Q4/Q3/Q2 quants)
huggingface-cli download mradermacher/SparkTTS-LLM-GGUF SparkTTS-LLM.f16.gguf --local-dir ./models/
huggingface-cli download mradermacher/SparkTTS-LLM-GGUF SparkTTS-LLM.Q8_0.gguf --local-dir ./models/
huggingface-cli download mradermacher/SparkTTS-LLM-GGUF SparkTTS-LLM.Q4_K_M.gguf --local-dir ./models/
huggingface-cli download mradermacher/SparkTTS-LLM-GGUF SparkTTS-LLM.Q3_K_L.gguf --local-dir ./models/
huggingface-cli download mradermacher/SparkTTS-LLM-GGUF SparkTTS-LLM.Q2_K.gguf --local-dir ./models/
"""

r"""
# ---- Inference Overview of Voice Cloning ----
python -m unittest test.modules.speech.tts.test_spark.TestSparkTTS.test_get_voices
REF_AUDIO_PATH=./test/audio_files/asr_example_zh.wav \
    REF_TEXT="欢迎大家来体验达摩院推出的语音识别模型" \
    python -m unittest test.modules.speech.tts.test_spark.TestSparkTTS.test_set_voice

REF_AUDIO_PATH=./test/audio_files/asr_example_zh.wav \
    REF_TEXT="欢迎大家来体验达摩院推出的语音识别模型" \
    TTS_TEXT="万物之始,大道至简,衍化至繁。君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。" \
    python -m unittest test.modules.speech.tts.test_spark.TestSparkTTS.test_synthesize

REF_AUDIO_PATH=./test/audio_files/asr_example_zh.wav \
    REF_TEXT="欢迎大家来体验达摩院推出的语音识别模型" \
    TTS_TEXT="万物之始,大道至简,衍化至繁。" \
    python -m unittest test.modules.speech.tts.test_spark.TestSparkTTS.test_synthesize

REF_AUDIO_PATH=./test/audio_files/asr_example_zh.wav \
    REF_TEXT="欢迎大家来体验达摩院推出的语音识别模型" \
    TTS_TEXT="万物之始,大道至简,衍化至繁。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。" \
    python -m unittest test.modules.speech.tts.test_spark.TestSparkTTS.test_synthesize_speak

# ---- Inference Overview of Controlled Generation with gender ----

python -m unittest test.modules.speech.tts.test_spark.TestSparkTTS.test_synthesize
python -m unittest test.modules.speech.tts.test_spark.TestSparkTTS.test_synthesize_speak

TTS_TEXT="万物之始,大道至简,衍化至繁。君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。" \
    python -m unittest test.modules.speech.tts.test_spark.TestSparkTTS.test_synthesize

TTS_TEXT="万物之始,大道至简,衍化至繁。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。" \
    python -m unittest test.modules.speech.tts.test_spark.TestSparkTTS.test_synthesize_speak

"""

""" ---- spark tranformers generator tts ----
TTS_TAG=tts_generator_spark \
    LLM_MODEL_NAME_OR_PATH=./models/SparkAudio/Spark-TTS-0.5B/LLM \
    python -m unittest test.modules.speech.tts.test_spark.TestSparkTTS.test_get_voices
TTS_TAG=tts_generator_spark \
    REF_AUDIO_PATH=./test/audio_files/asr_example_zh.wav \
    REF_TEXT="欢迎大家来体验达摩院推出的语音识别模型" \
    python -m unittest test.modules.speech.tts.test_spark.TestSparkTTS.test_set_voice

TTS_TAG=tts_generator_spark \
    LLM_MODEL_NAME_OR_PATH=./models/SparkAudio/Spark-TTS-0.5B/LLM \
    REF_AUDIO_PATH=./test/audio_files/asr_example_zh.wav \
    REF_TEXT="欢迎大家来体验达摩院推出的语音识别模型" \
    TTS_TEXT="万物之始,大道至简,衍化至繁。君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。" \
    python -m unittest test.modules.speech.tts.test_spark.TestSparkTTS.test_synthesize

TTS_TAG=tts_generator_spark \
    LLM_MODEL_NAME_OR_PATH=./models/SparkAudio/Spark-TTS-0.5B/LLM \
    REF_AUDIO_PATH=./test/audio_files/asr_example_zh.wav \
    REF_TEXT="欢迎大家来体验达摩院推出的语音识别模型" \
    TTS_TEXT="万物之始,大道至简,衍化至繁。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。" \
    python -m unittest test.modules.speech.tts.test_spark.TestSparkTTS.test_synthesize_speak

# ---- Inference Overview of Controlled Generation with gender ----

TTS_TAG=tts_generator_spark \
    LLM_MODEL_NAME_OR_PATH=./models/SparkAudio/Spark-TTS-0.5B/LLM \
    python -m unittest test.modules.speech.tts.test_spark.TestSparkTTS.test_synthesize
TTS_TAG=tts_generator_spark \
    LLM_MODEL_NAME_OR_PATH=./models/SparkAudio/Spark-TTS-0.5B/LLM \
    python -m unittest test.modules.speech.tts.test_spark.TestSparkTTS.test_synthesize_speak

TTS_TAG=tts_generator_spark \
    LLM_MODEL_NAME_OR_PATH=./models/SparkAudio/Spark-TTS-0.5B/LLM \
    TTS_TEXT="万物之始,大道至简,衍化至繁。君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。" \
    python -m unittest test.modules.speech.tts.test_spark.TestSparkTTS.test_synthesize

TTS_TAG=tts_generator_spark \
    LLM_MODEL_NAME_OR_PATH=./models/SparkAudio/Spark-TTS-0.5B/LLM \
    TTS_TEXT="万物之始,大道至简,衍化至繁。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。" \
    python -m unittest test.modules.speech.tts.test_spark.TestSparkTTS.test_synthesize_speak
"""

""" ---- spark llamacpp generatro tts ----
TTS_TAG=tts_generator_spark \
    TTS_LM_GENERATOR_TAG=llm_llamacpp_generator \
    LLM_MODEL_PATH=./models/SparkTTS-LLM.Q8_0.gguf \
    python -m unittest test.modules.speech.tts.test_spark.TestSparkTTS.test_get_voices
TTS_TAG=tts_generator_spark \
    TTS_LM_GENERATOR_TAG=llm_llamacpp_generator \
    LLM_MODEL_PATH=./models/SparkTTS-LLM.Q8_0.gguf \
    REF_AUDIO_PATH=./test/audio_files/asr_example_zh.wav \
    REF_TEXT="欢迎大家来体验达摩院推出的语音识别模型" \
    python -m unittest test.modules.speech.tts.test_spark.TestSparkTTS.test_set_voice

TTS_TAG=tts_generator_spark \
    TTS_LM_GENERATOR_TAG=llm_llamacpp_generator \
    LLM_MODEL_PATH=./models/SparkTTS-LLM.Q2_K.gguf \
    REF_AUDIO_PATH=./test/audio_files/asr_example_zh.wav \
    REF_TEXT="欢迎大家来体验达摩院推出的语音识别模型" \
    TTS_TEXT="万物之始,大道至简,衍化至繁。君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。" \
    python -m unittest test.modules.speech.tts.test_spark.TestSparkTTS.test_synthesize

TTS_TAG=tts_generator_spark \
    TTS_LM_GENERATOR_TAG=llm_llamacpp_generator \
    LLM_MODEL_PATH=./models/SparkTTS-LLM.Q3_K_L.gguf \
    REF_AUDIO_PATH=./test/audio_files/asr_example_zh.wav \
    REF_TEXT="欢迎大家来体验达摩院推出的语音识别模型" \
    TTS_TEXT="万物之始,大道至简,衍化至繁。君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。" \
    python -m unittest test.modules.speech.tts.test_spark.TestSparkTTS.test_synthesize

TTS_TAG=tts_generator_spark \
    TTS_LM_GENERATOR_TAG=llm_llamacpp_generator \
    LLM_MODEL_PATH=./models/SparkTTS-LLM.Q4_K_M.gguf \
    REF_AUDIO_PATH=./test/audio_files/asr_example_zh.wav \
    REF_TEXT="欢迎大家来体验达摩院推出的语音识别模型" \
    TTS_TEXT="万物之始,大道至简,衍化至繁。君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。" \
    python -m unittest test.modules.speech.tts.test_spark.TestSparkTTS.test_synthesize

TTS_TAG=tts_generator_spark \
    TTS_LM_GENERATOR_TAG=llm_llamacpp_generator \
    LLM_MODEL_PATH=./models/SparkTTS-LLM.Q8_0.gguf \
    REF_AUDIO_PATH=./test/audio_files/asr_example_zh.wav \
    REF_TEXT="欢迎大家来体验达摩院推出的语音识别模型" \
    TTS_TEXT="万物之始,大道至简,衍化至繁。君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。" \
    python -m unittest test.modules.speech.tts.test_spark.TestSparkTTS.test_synthesize

TTS_TAG=tts_generator_spark \
    TTS_LM_GENERATOR_TAG=llm_llamacpp_generator \
    LLM_MODEL_PATH=./models/SparkTTS-LLM.f16.gguf \
    REF_AUDIO_PATH=./test/audio_files/asr_example_zh.wav \
    REF_TEXT="欢迎大家来体验达摩院推出的语音识别模型" \
    TTS_TEXT="万物之始,大道至简,衍化至繁。君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。" \
    python -m unittest test.modules.speech.tts.test_spark.TestSparkTTS.test_synthesize

TTS_TAG=tts_generator_spark \
    TTS_LM_GENERATOR_TAG=llm_llamacpp_generator \
    LLM_MODEL_PATH=./models/SparkTTS-LLM.Q8_0.gguf \
    REF_AUDIO_PATH=./test/audio_files/asr_example_zh.wav \
    REF_TEXT="欢迎大家来体验达摩院推出的语音识别模型" \
    TTS_TEXT="万物之始,大道至简,衍化至繁。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。" \
    python -m unittest test.modules.speech.tts.test_spark.TestSparkTTS.test_synthesize_speak

# ---- Inference Overview of Controlled Generation with gender ----

TTS_TAG=tts_generator_spark \
    TTS_LM_GENERATOR_TAG=llm_llamacpp_generator \
    LLM_MODEL_PATH=./models/SparkTTS-LLM.Q8_0.gguf \
    python -m unittest test.modules.speech.tts.test_spark.TestSparkTTS.test_synthesize
TTS_TAG=tts_generator_spark \
    TTS_LM_GENERATOR_TAG=llm_llamacpp_generator \
    LLM_MODEL_PATH=./models/SparkTTS-LLM.Q8_0.gguf \
    python -m unittest test.modules.speech.tts.test_spark.TestSparkTTS.test_synthesize_speak

TTS_TAG=tts_generator_spark \
    TTS_LM_GENERATOR_TAG=llm_llamacpp_generator \
    LLM_MODEL_PATH=./models/SparkTTS-LLM.Q8_0.gguf \
    TTS_TEXT="万物之始,大道至简,衍化至繁。君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。" \
    python -m unittest test.modules.speech.tts.test_spark.TestSparkTTS.test_synthesize

TTS_TAG=tts_generator_spark \
    TTS_LM_GENERATOR_TAG=llm_llamacpp_generator \
    LLM_MODEL_PATH=./models/SparkTTS-LLM.Q8_0.gguf \
    TTS_TEXT="万物之始,大道至简,衍化至繁。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。" \
    python -m unittest test.modules.speech.tts.test_spark.TestSparkTTS.test_synthesize_speak
"""


class TestSparkTTS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tts_tag = os.getenv("TTS_TAG", "tts_spark")
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

        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)

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
        self.session.ctx.state["temperature"] = 0.1
        print(self.session.ctx)
        iter = self.tts.synthesize_sync(self.session)
        res = bytearray()
        times = []
        start_time = perf_counter()
        for i, chunk in enumerate(iter):
            print(i, len(chunk))
            res.extend(chunk)
            times.append(perf_counter() - start_time)
            start_time = perf_counter()
        print(
            f"generate fisrt chunk({len(chunk)}) time: {times[0]} s, {len(res)} waveform cost time: {sum(times)} s, avg cost time: {sum(times)/len(res)}"
        )

        stream_info = self.tts.get_stream_info()
        print(f"stream_info:{stream_info}")

        file_name = f"test_{self.tts.TAG}.wav"
        os.makedirs(RECORDS_DIR, exist_ok=True)
        file_path = os.path.join(RECORDS_DIR, file_name)
        data = np.frombuffer(res, dtype=stream_info["np_dtype"])

        # 去除静音部分
        data, _ = librosa.effects.trim(data, top_db=60)

        soundfile.write(file_path, data, stream_info["rate"])
        info = soundfile.info(file_path, verbose=True)
        print(info)

        print(
            f"tts cost time {sum(times)} s, wav duration {info.duration} s, RTF: {sum(times)/info.duration}"
        )

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
