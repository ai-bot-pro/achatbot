import os
import asyncio
import logging

import unittest

from src.common.utils.helper import get_audio_segment
from src.common.utils.wav import save_audio_to_file
from src.common.logger import Logger
from src.common.factory import EngineFactory, EngineClass
from src.common.session import Session
from src.common.types import SessionCtx, TEST_DIR, MODELS_DIR, RECORDS_DIR, SileroVADArgs
from src.common.interface import IDetector
import src.modules.speech.detector.silero_vad

r"""
CHECK_FRAMES_MODE=0 python -m unittest test.modules.speech.detector.test_silero_vad.TestSileroVADDetector.test_detect
CHECK_FRAMES_MODE=1 python -m unittest test.modules.speech.detector.test_silero_vad.TestSileroVADDetector.test_detect
CHECK_FRAMES_MODE=2 python -m unittest test.modules.speech.detector.test_silero_vad.TestSileroVADDetector.test_detect

SILERO_REPO_OR_DIR=/Users/wuyong/.cache/torch/hub/snakers4_silero-vad_master \
    SILERO_MODEL_SOURCE=local \
    python -m unittest test.modules.speech.detector.test_silero_vad.TestSileroVADDetector.test_detect
"""


class TestSileroVADDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tag = os.getenv("DETECTOR_TAG", "silero_vad")
        audio_file = os.path.join(TEST_DIR, "audio_files", "vad_test.wav")
        cls.audio_file = os.getenv("AUDIO_FILE", audio_file)
        cls.repo_or_dir = os.getenv("SILERO_REPO_OR_DIR", "snakers4/silero-vad")
        cls.model = os.getenv("SILERO_MODEL", "silero_vad")
        cls.check_frames_mode = int(os.getenv("CHECK_FRAMES_MODE", "1"))
        cls.source = os.getenv("SILERO_MODEL_SOURCE", "github")
        cls.trust_repo = bool(os.getenv("SILERO_TRUST_REPO", "1"))

        Logger.init(os.getenv("LOG_LEVEL", "debug").upper(), is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        kwargs = SileroVADArgs(
            repo_or_dir=self.repo_or_dir,
            model=self.model,
            source=self.source,
            trust_repo=self.trust_repo,
            check_frames_mode=self.check_frames_mode,
        ).__dict__
        print(kwargs)
        self.detector: IDetector | EngineClass = EngineFactory.get_engine_by_tag(
            EngineClass, self.tag, **kwargs
        )
        self.session = Session(**SessionCtx("test_silero_vad_client_id", 16000, 2).__dict__)

    def tearDown(self):
        self.detector.close()

    def test_detect(self):
        audio_segment = asyncio.run(get_audio_segment(self.audio_file))
        print(audio_segment.frame_rate)
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
        paud = pyaudio.PyAudio()
        audio_stream = paud.open(
            rate=rate, channels=1, format=pyaudio.paInt16, input=True, frames_per_buffer=1024
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
