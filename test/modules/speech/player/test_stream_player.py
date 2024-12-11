import os
import logging
import asyncio

import unittest

from src.modules.speech.audio_stream import AudioStreamEnvInit
from src.modules.speech.player import PlayerEnvInit
from src.common.interface import IAudioStream
from src.modules.speech.player.stream_player import StreamPlayer
from src.common.logger import Logger
from src.common.factory import EngineClass
from src.common.session import Session
from src.common.utils import helper
from src.common.types import SessionCtx, TEST_DIR
from src.types.speech.audio_stream import AudioStreamInfo

r"""

TTS_TAG=tts_16k_speaker \
    python -m unittest test.modules.speech.player.test_stream_player.TestStreamPlayer.test_play_audio

# need create daily room, get room_url
AUDIO_OUT_STREAM_TAG=daily_room_audio_out_stream \
    MEETING_ROOM_URL=https://weedge.daily.co/chat-bot \
    TTS_TAG=tts_16k_speaker \
    python -m unittest test.modules.speech.player.test_stream_player.TestStreamPlayer.test_play_audio
"""


class TestStreamPlayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tag = os.getenv("PLAYER_TAG", "stream_player")
        Logger.init(os.getenv("LOG_LEVEL", "debug").upper(), is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.player: StreamPlayer = PlayerEnvInit.initPlayerEngine()
        print(self.player.args.__dict__)
        self.audio_out_stream: IAudioStream | EngineClass = (
            AudioStreamEnvInit.initAudioOutStreamEngine()
        )
        self.player.set_out_stream(self.audio_out_stream)
        self.annotations_path = os.path.join(TEST_DIR, "audio_files/annotations.json")
        self.session = Session(**SessionCtx("test_client_id").__dict__)
        self.out_stream_info: AudioStreamInfo = self.audio_out_stream.get_stream_info()

    def tearDown(self):
        self.player.close()

    def test_play_audio(self):
        self.player.open()
        annotations = asyncio.run(helper.load_json(self.annotations_path))
        for audio_file, data in annotations.items():
            audio_file_path = os.path.join(TEST_DIR, f"audio_files/{audio_file}")
            for segment in data["segments"]:
                self.player.start(self.session)
                audio_segment = asyncio.run(
                    helper.get_audio_segment(audio_file_path, segment["start"], segment["end"])
                )
                print(
                    f"channels:{audio_segment.channels},sample_width:{audio_segment.sample_width},frame_rate:{audio_segment.frame_rate},frame_width:{audio_segment.frame_width}"
                )
                self.assertEqual(audio_segment.frame_rate, self.out_stream_info.out_sample_rate)
                self.assertEqual(audio_segment.channels, self.out_stream_info.out_channels)
                self.assertEqual(audio_segment.sample_width, self.out_stream_info.out_sample_width)
                self.session.ctx.state["tts_chunk"] = audio_segment.raw_data
                logging.debug(f"chunk size: {len(audio_segment.raw_data)}")
                self.player.play_audio(self.session)
                self.player.stop(self.session)
