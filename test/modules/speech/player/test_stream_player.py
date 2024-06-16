import os
import logging
import asyncio

import unittest

from src.common.logger import Logger
from src.common.factory import EngineFactory, EngineClass
from src.common.session import Session
from src.common.utils import audio_utils
from src.common.types import SessionCtx, TEST_DIR, MODELS_DIR, RECORDS_DIR, CHUNK
import src.modules.speech

r"""
python -m unittest test.modules.speech.player.test_stream_player.TestStreamPlayer.test_play_audio
"""


class TestStreamPlayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tag = os.getenv('PLAYER_TAG', "stream_player")
        Logger.init(logging.DEBUG)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        kwargs = {}
        kwargs["chunk_size"] = CHUNK
        self.player = EngineFactory.get_engine_by_tag(
            EngineClass, self.tag, **kwargs)
        self.annotations_path = os.path.join(
            TEST_DIR, "audio_files/annotations.json")
        self.session = Session(**SessionCtx(
            "test_client_id").__dict__)

    def tearDown(self):
        self.player.close()

    def test_play_audio(self):
        annotations = asyncio.run(audio_utils.load_json(self.annotations_path))
        for audio_file, data in annotations.items():
            audio_file_path = os.path.join(
                TEST_DIR, f"audio_files/{audio_file}")
            for segment in data["segments"]:
                audio_segment = asyncio.run(audio_utils.get_audio_segment(
                    audio_file_path,
                    segment["start"],
                    segment["end"]))
                self.session.ctx.state["tts_chunk"] = audio_segment.raw_data
                logging.debug(f"chunk size: {len(audio_segment.raw_data)}")
                self.player.play_audio(self.session)
