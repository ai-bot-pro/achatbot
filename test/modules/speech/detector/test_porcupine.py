import collections
import os
import asyncio
import logging

import unittest

from src.common.utils.helper import get_audio_segment
from src.common.logger import Logger
from src.common.factory import EngineFactory, EngineClass
from src.common.session import Session
from src.common.types import SessionCtx, MODELS_DIR, RECORDS_DIR
from src.common.interface import IDetector
import src.modules.speech.detector.porcupine

r"""
python -m unittest test.modules.speech.detector.test_porcupine.TestPorcupineWakeWordDetector.test_detect
"""


class TestPorcupineWakeWordDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tag = os.getenv("DETECTOR_TAG", "porcupine_wakeword")
        cls.wake_words = os.getenv("WAKE_WORDS", "小黑")
        audio_file = os.path.join(RECORDS_DIR, "tmp_wakeword_porcupine.wav")
        model_path = os.path.join(MODELS_DIR, "porcupine_params_zh.pv")
        keyword_paths = os.path.join(MODELS_DIR, "小黑_zh_mac_v3_0_0.ppn")
        cls.audio_file = os.getenv("AUDIO_FILE", audio_file)
        cls.model_path = os.getenv("MODEL_PATH", model_path)
        cls.keyword_paths = os.getenv("KEYWORD_PATHS", keyword_paths)

        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        kwargs = {}
        kwargs["wake_words"] = self.wake_words
        kwargs["model_path"] = self.model_path
        kwargs["keyword_paths"] = self.keyword_paths.split(",")
        print(kwargs)
        self.detector: IDetector = EngineFactory.get_engine_by_tag(EngineClass, self.tag, **kwargs)
        self.session = Session(**SessionCtx("test_client_id", 16000, 2).__dict__)

        sample_rate, frame_length = self.detector.get_sample_info()
        print(sample_rate, frame_length)
        pre_recording_buffer_duration = 3.0
        maxlen = int((sample_rate // frame_length) * pre_recording_buffer_duration)
        print(f"audio_buffer maxlen: {maxlen}")
        # ring buffer
        self.audio_buffer = collections.deque(maxlen=maxlen)
        self.sample_rate, self.frame_length = sample_rate, frame_length

    def tearDown(self):
        self.detector.close()

    def test_detect(self):
        audio_segment = asyncio.run(get_audio_segment(self.audio_file))
        self.assertEqual(audio_segment.frame_rate, 16000)
        self.audio_buffer.append(audio_segment.raw_data)
        self.session.ctx.read_audio_frames = audio_segment.raw_data
        self.detector.set_audio_data(self.audio_buffer)
        res = asyncio.run(self.detector.detect(self.session))
        logging.debug(res)
        self.assertEqual(res, False)

    def test_record_detect(self):
        import pyaudio

        paud = pyaudio.PyAudio()
        audio_stream = paud.open(
            rate=self.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=1024,
        )

        audio_stream.start_stream()
        logging.debug("start recording")
        while True:
            read_audio_frames = audio_stream.read(self.frame_length)
            self.session.ctx.read_audio_frames = read_audio_frames
            self.detector.set_audio_data(self.audio_buffer)
            res = asyncio.run(self.detector.detect(self.session))
            logging.debug(res)
            if res is True:
                break
            self.audio_buffer.append(read_audio_frames)

        audio_stream.stop_stream()
        audio_stream.close()
        paud.terminate()
