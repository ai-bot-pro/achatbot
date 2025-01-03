import os
import asyncio
import logging

import unittest

from src.common.utils.audio_utils import convert_sampling_rate_to_16k
from src.common.utils.helper import get_audio_segment
from src.common.utils.wav import save_audio_to_file, read_audio_file
from src.common.logger import Logger
from src.common.factory import EngineFactory, EngineClass
from src.common.session import Session
from src.common.types import SessionCtx, TEST_DIR, MODELS_DIR, RECORDS_DIR, FSMNVADArgs
from src.common.interface import IDetector
import src.modules.speech.detector.fsmn_vad

r"""
CHECK_FRAMES_MODE=0 python -m unittest test.modules.speech.detector.test_fsmn_vad.TestFSMNVADDetector.test_detect
CHECK_FRAMES_MODE=1 python -m unittest test.modules.speech.detector.test_fsmn_vad.TestFSMNVADDetector.test_detect
CHECK_FRAMES_MODE=2 python -m unittest test.modules.speech.detector.test_fsmn_vad.TestFSMNVADDetector.test_detect

# use pyaudio to record audio, so need say something to vad, e.g.: say "hello"
CHECK_FRAMES_MODE=1 python -m unittest test.modules.speech.detector.test_fsmn_vad.TestFSMNVADDetector.test_record_detect
"""


class TestFSMNVADDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tag = os.getenv("DETECTOR_TAG", "fsmn_vad")
        audio_file = os.path.join(TEST_DIR, "audio_files", "vad_test.wav")
        cls.audio_file = os.getenv("AUDIO_FILE", audio_file)
        cls.check_frames_mode = int(os.getenv("CHECK_FRAMES_MODE", "1"))

        Logger.init(os.getenv("LOG_LEVEL", "debug").upper(), is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        kwargs = FSMNVADArgs(
            check_frames_mode=self.check_frames_mode,
        ).__dict__
        print(kwargs)
        self.detector: IDetector | EngineClass = EngineFactory.get_engine_by_tag(
            EngineClass, self.tag, **kwargs
        )
        self.session = Session(**SessionCtx("test_fsmn_vad_client_id", 16000, 2).__dict__)

    def tearDown(self):
        self.detector.close()

    def test_convert_sampling_rate_to_16k(self):
        convert_sampling_rate_to_16k(
            self.audio_file, os.path.join(RECORDS_DIR, "test_convert_sampling_rate_to_16k.wav")
        )

    def test_detect(self):
        audio_segment = asyncio.run(get_audio_segment(self.audio_file))
        print("audio_segment-->", audio_segment.frame_rate, len(audio_segment.raw_data))
        self.detector.set_args(sample_rate=audio_segment.frame_rate)
        print(self.detector.args)
        self.assertEqual(self.detector.args.sample_rate, audio_segment.frame_rate)
        self.detector.set_audio_data(audio_segment.raw_data)
        if hasattr(self.detector, "audio_buffer"):
            file_path = asyncio.run(
                save_audio_to_file(
                    self.detector.audio_buffer,
                    self.session.get_record_audio_name(),
                    audio_dir=RECORDS_DIR,
                )
            )
            print(file_path)
        res = asyncio.run(self.detector.detect(self.session))
        logging.debug(res)
        if self.check_frames_mode == 1:
            self.assertEqual(res, True)
        else:
            self.assertEqual(res, False)

    def test_record_detect(self):
        import pyaudio

        rate, frame_len = self.detector.get_sample_info()
        print(rate, frame_len)
        paud = pyaudio.PyAudio()
        audio_stream = paud.open(
            rate=rate, channels=1, format=pyaudio.paInt16, input=True, frames_per_buffer=frame_len
        )

        audio_stream.start_stream()
        logging.debug("start recording")
        while True:
            read_audio_frames = audio_stream.read(frame_len)
            self.detector.set_audio_data(read_audio_frames)
            res = asyncio.run(self.detector.detect(self.session))
            logging.debug(res)
            if res is True:
                break

        audio_stream.stop_stream()
        audio_stream.close()
        paud.terminate()
