import os
import logging
import asyncio

import unittest

from src.common.logger import Logger
from src.common.factory import EngineFactory
from src.common.session import Session
from src.common.utils import audio_utils
from src.common.types import SessionCtx, TEST_DIR, MODELS_DIR, RECORDS_DIR, INT16_MAX_ABS_VALUE
from src.modules.speech.recorder.record import EngineClass

r"""
"""


class TestRMSRecorder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tag = os.getenv('RECODER_TAG', "rms_recorder")
        cls.input_device_index = os.getenv('MIC_IDX', "1")
        Logger.init(logging.DEBUG)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        kwargs = {}
        kwargs["input_device_index"] = int(self.input_device_index)
        self.recorder = EngineFactory.get_engine_by_tag(
            EngineClass, self.tag, **kwargs)
        self.session = Session(**SessionCtx(
            "test_client_id").__dict__)
        pass

    def tearDown(self):
        self.recorder.close()
        pass

    def test_record(self):
        frames = self.recorder.record_audio(self.session)
        self.assertGreater(len(frames), 0)
        asyncio.run(audio_utils.save_audio_to_file(
            frames, os.path.join(RECORDS_DIR, "test.wav")))
