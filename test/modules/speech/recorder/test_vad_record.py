import os
import logging
import asyncio

import unittest

from src.modules.speech.audio_stream import AudioStreamEnvInit
from src.modules.speech.detector import VADEnvInit
from src.modules.speech.recorder import RecorderEnvInit
from src.common.interface import IAudioStream, IRecorder
from src.common.logger import Logger
from src.common.factory import EngineFactory, EngineClass
from src.common.session import Session
from src.common.utils import wav
from src.common.types import SessionCtx, MODELS_DIR, RECORDS_DIR

import src.modules.speech.detector.porcupine

r"""
python -m unittest test.modules.speech.recorder.test_vad_record.TestVADRecorder.test_record
IS_STREAM_CALLBACK=1 python -m unittest test.modules.speech.recorder.test_vad_record.TestVADRecorder.test_record

IS_PAD_TENSOR=1 VAD_DETECTOR_TAG=silero_vad python -m unittest test.modules.speech.recorder.test_vad_record.TestVADRecorder.test_record
IS_PAD_TENSOR=1 IS_STREAM_CALLBACK=1 VAD_DETECTOR_TAG=silero_vad python -m unittest test.modules.speech.recorder.test_vad_record.TestVADRecorder.test_record

IS_PAD_TENSOR=1 VAD_DETECTOR_TAG=webrtc_silero_vad python -m unittest test.modules.speech.recorder.test_vad_record.TestVADRecorder.test_record
IS_PAD_TENSOR=1 IS_STREAM_CALLBACK=1 VAD_DETECTOR_TAG=webrtc_silero_vad python -m unittest test.modules.speech.recorder.test_vad_record.TestVADRecorder.test_record

# need create daily room, get room_url
AUDIO_IN_STREAM_TAG=daily_room_audio_in_stream \
    MEETING_ROOM_URL=https://weedge.daily.co/chat-bot \
    RECODER_TAG=vad_recorder \
    IS_PAD_TENSOR=1 VAD_DETECTOR_TAG=webrtc_silero_vad \
    python -m unittest test.modules.speech.recorder.test_vad_record.TestVADRecorder.test_record
AUDIO_IN_STREAM_TAG=daily_room_audio_in_stream \
    MEETING_ROOM_URL=https://weedge.daily.co/chat-bot \
    IS_STREAM_CALLBACK=1 RECODER_TAG=vad_recorder \
    IS_PAD_TENSOR=1 VAD_DETECTOR_TAG=webrtc_silero_vad \
    python -m unittest test.modules.speech.recorder.test_vad_record.TestVADRecorder.test_record

RECODER_TAG=wakeword_vad_recorder python -m unittest test.modules.speech.recorder.test_vad_record.TestVADRecorder.test_wakeword_record
IS_STREAM_CALLBACK=1 RECODER_TAG=wakeword_vad_recorder python -m unittest test.modules.speech.recorder.test_vad_record.TestVADRecorder.test_wakeword_record

IS_PAD_TENSOR=1 VAD_DETECTOR_TAG=silero_vad RECODER_TAG=wakeword_vad_recorder python -m unittest test.modules.speech.recorder.test_vad_record.TestVADRecorder.test_wakeword_record
IS_PAD_TENSOR=1 IS_STREAM_CALLBACK=1 VAD_DETECTOR_TAG=silero_vad RECODER_TAG=wakeword_vad_recorder python -m unittest test.modules.speech.recorder.test_vad_record.TestVADRecorder.test_wakeword_record

VAD_DETECTOR_TAG=webrtc_silero_vad RECODER_TAG=wakeword_vad_recorder python -m unittest test.modules.speech.recorder.test_vad_record.TestVADRecorder.test_wakeword_record
IS_PAD_TENSOR=1 IS_STREAM_CALLBACK=1 VAD_DETECTOR_TAG=webrtc_silero_vad RECODER_TAG=wakeword_vad_recorder python -m unittest test.modules.speech.recorder.test_vad_record.TestVADRecorder.test_wakeword_record
"""


class TestVADRecorder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector_wake_tag = os.getenv("DETECTOR_WAKE_TAG", "porcupine_wakeword")
        cls.wake_words = os.getenv("WAKE_WORDS", "小黑")
        audio_file = os.path.join(RECORDS_DIR, "tmp_wakeword_porcupine.wav")
        model_path = os.path.join(MODELS_DIR, "porcupine_params_zh.pv")
        keyword_paths = os.path.join(MODELS_DIR, "小黑_zh_mac_v3_0_0.ppn")
        cls.audio_file = os.getenv("AUDIO_FILE", audio_file)
        cls.model_path = os.getenv("MODEL_PATH", model_path)
        cls.keyword_paths = os.getenv("KEYWORD_PATHS", keyword_paths)

        Logger.init(os.getenv("LOG_LEVEL", "debug").upper(), is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        os.environ["RECORDER_TAG"] = "vad_recorder"
        os.environ["IS_STREAM_CALLBACK"] = os.getenv("IS_STREAM_CALLBACK", "")
        self.recorder: IRecorder | EngineClass = RecorderEnvInit.initRecorderEngine()
        self.audio_in_stream: IAudioStream | EngineClass = (
            AudioStreamEnvInit.initAudioInStreamEngine()
        )
        self.recorder.set_in_stream(self.audio_in_stream)
        self.recorder.open()
        self.session = Session(**SessionCtx("test_client_id").__dict__)

        self.session.ctx.vad = VADEnvInit.initVADEngine()

    def tearDown(self):
        self.recorder and self.recorder.close()
        self.session.close()

    def test_record(self):
        frames = asyncio.run(self.recorder.record_audio(self.session))
        self.assertGreater(len(frames), 0)
        data = b"".join(frames)
        file_path = asyncio.run(wav.save_audio_to_file(data, os.path.join(RECORDS_DIR, "test.wav")))
        print(file_path)

    def test_multi_record(self):
        frames = asyncio.run(self.recorder.record_audio(self.session))
        self.assertGreater(len(frames), 0)
        data = b"".join(frames)
        file_path = asyncio.run(wav.save_audio_to_file(data, os.path.join(RECORDS_DIR, "test.wav")))
        print(file_path)

        self.recorder2: IRecorder | EngineClass = RecorderEnvInit.initRecorderEngine()
        self.recorder2.set_in_stream(self.audio_in_stream)
        self.recorder2.open()
        frames = asyncio.run(self.recorder2.record_audio(self.session))
        self.assertGreater(len(frames), 0)
        data = b"".join(frames)
        file_path = asyncio.run(
            wav.save_audio_to_file(data, os.path.join(RECORDS_DIR, "test2.wav"))
        )
        print(file_path)
        self.recorder2.close()

    def test_wakeword_record(self):
        def on_wakeword_detected(session, data):
            print(
                f"bot_name:{session.ctx.state['bot_name']} wakeword detected, data_len:{len(data)}"
            )

        kwargs = {}
        kwargs["wake_words"] = self.wake_words
        kwargs["model_path"] = self.model_path
        # kwargs["on_wakeword_detected"] = on_wakeword_detected
        kwargs["keyword_paths"] = self.keyword_paths.split(",")
        self.session.ctx.waker = EngineFactory.get_engine_by_tag(
            EngineClass, self.detector_wake_tag, **kwargs
        )
        self.session.ctx.waker.set_args(on_wakeword_detected=on_wakeword_detected)

        round = 1
        for i in range(round):
            frames = asyncio.run(self.recorder.record_audio(self.session))
            self.assertGreaterEqual(len(frames), 0)
            data = b"".join(frames)
            file_path = asyncio.run(
                wav.save_audio_to_file(data, os.path.join(RECORDS_DIR, f"test{i}.wav"))
            )
            print(file_path)
