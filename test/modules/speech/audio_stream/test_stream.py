
import logging
import os
import time
import asyncio
import queue

import unittest

from src.common.factory import EngineClass
from src.common.utils.audio_utils import get_audio_segment
from src.common.session import Session
from src.common.types import TEST_DIR, SessionCtx
from src.common.logger import Logger
from src.common.interface import IAudioStream
from src.cmd.init import Env

"""
AUDIO_IN_STREAM_TAG=pyaudio_in_stream \
    python -m unittest test.modules.speech.audio_stream.test_stream.TestAudioInStream
AUDIO_IN_STREAM_TAG=pyaudio_in_stream \
    IS_CALLBACK=1 \
    python -m unittest test.modules.speech.audio_stream.test_stream.TestAudioInStream

# need create daily room, get room_url
AUDIO_IN_STREAM_TAG=daily_room_audio_in_stream \
    MEETING_ROOM_URL=https://weedge.daily.co/chat-bot \
    python -m unittest test.modules.speech.audio_stream.test_stream.TestAudioInStream
AUDIO_IN_STREAM_TAG=daily_room_audio_in_stream \
    IS_CALLBACK=1 \
    MEETING_ROOM_URL=https://weedge.daily.co/chat-bot \
    python -m unittest test.modules.speech.audio_stream.test_stream.TestAudioInStream
"""


class TestAudioInStream(unittest.TestCase):
    buffer_queue = queue.Queue()

    def stream_callback(self, in_data, frame_count=0, time_info=None, status=0):
        print(len(in_data), frame_count, time_info, status)
        self.assertEqual(frame_count, self.stream_info["frames_per_buffer"])
        self.assertEqual(
            len(in_data),
            self.stream_info["in_sample_width"] *
            self.stream_info["in_channels"] *
            self.stream_info["frames_per_buffer"]
        )
        self.buffer_queue and self.buffer_queue.put_nowait(in_data)
        play_data = chr(0) * len(in_data)
        return play_data, 0

    def get_record_buf(self, num_frames) -> bytes:
        if self.is_callback is False:
            return self.audio_in_stream.read_stream(num_frames)
        return self.buffer_queue.get()

    @classmethod
    def setUpClass(cls):
        cls.tag = os.getenv('AUDIO_IN_STREAM_TAG', "pyaudio_in_stream")
        cls.is_callback = bool(os.getenv('IS_CALLBACK', ""))
        Logger.init(logging.DEBUG, is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.audio_in_stream: IAudioStream | EngineClass = Env.initAudioInStreamEngine()
        self.stream_info = self.audio_in_stream.get_stream_info()
        self.session = Session(**SessionCtx(
            "test_client_id").__dict__)
        if self.is_callback:
            self.audio_in_stream.set_args(stream_callback=self.stream_callback)

    def tearDown(self):
        self.audio_in_stream.close()

    def test_in(self):
        is_stream_active = self.audio_in_stream.is_stream_active()
        self.assertEqual(is_stream_active, False)
        self.audio_in_stream.open_stream()
        self.audio_in_stream.start_stream()
        is_stream_active = self.audio_in_stream.is_stream_active()
        self.assertEqual(is_stream_active, True)

        read_cn = 10
        while read_cn > 0:
            # 100 ms num_frames
            num_frames = int(self.stream_info["in_sample_rate"] / 10)
            data = self.get_record_buf(num_frames)
            print(len(data))
            if self.is_callback is False:
                self.assertEqual(len(data), num_frames * self.stream_info["in_sample_width"])
            read_cn -= 1

        self.audio_in_stream.stop_stream()
        self.audio_in_stream.close_stream()


"""
AUDIO_OUT_STREAM_TAG=pyaudio_out_stream \
    TTS_TAG=tts_16k_speaker \
    python -m unittest test.modules.speech.audio_stream.test_stream.TestAudioOutStream

AUDIO_OUT_STREAM_TAG=daily_room_audio_out_stream \
    MEETING_ROOM_URL=https://weedge.daily.co/chat-bot \
    TTS_TAG=tts_16k_speaker \
    python -m unittest test.modules.speech.audio_stream.test_stream.TestAudioOutStream
"""


class TestAudioOutStream(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        audio_file = os.path.join(TEST_DIR, f"audio_files", f"vad_test.wav")
        cls.audio_file = os.getenv('AUDIO_FILE', audio_file)
        Logger.init(logging.DEBUG, is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.audio_out_stream = Env.initAudioOutStreamEngine()
        self.session = Session(**SessionCtx("test_client_id").__dict__)

    def tearDown(self):
        self.audio_out_stream.close()

    def test_out(self):
        audio_segment = asyncio.run(get_audio_segment(self.audio_file))
        info = self.audio_out_stream.get_stream_info()
        print(info)
        self.assertEqual(info["out_sample_rate"], audio_segment.frame_rate)
        self.audio_out_stream.open_stream()
        self.audio_out_stream.start_stream()
        if len(audio_segment.raw_data) > 0:
            self.audio_out_stream.write_stream(audio_segment.raw_data)
        else:
            time.sleep(0.3)  # sleep for daily client -> release
        self.audio_out_stream.stop_stream()
        self.audio_out_stream.close_stream()
