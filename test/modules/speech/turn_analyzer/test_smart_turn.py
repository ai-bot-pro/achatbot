import logging
import unittest
import os

import wave
from pydub import AudioSegment

from src.common.logger import Logger
from src.common.session import Session
from src.common.interface import EndOfTurnState, ITurnAnalyzer
from src.common.types import SessionCtx, TEST_DIR, MODELS_DIR, RECORDS_DIR
from src.modules.speech.turn_analyzer import TurnAnalyzerEnvInit


r"""
LOG_LEVEL=debug TURN_ANALYZER_TAG=v2_smart_turn_analyzer python -m unittest test.modules.speech.turn_analyzer.test_smart_turn.TestTurnAnalyzer.test_end
TURN_TORCH_DTYPE=bfloat16 TURN_ANALYZER_TAG=v2_smart_turn_analyzer python -m unittest test.modules.speech.turn_analyzer.test_smart_turn.TestTurnAnalyzer.test_end

TURN_ANALYZER_TAG=v2_smart_turn_analyzer python -m unittest test.modules.speech.turn_analyzer.test_smart_turn.TestTurnAnalyzer.test_no_end
"""


class TestTurnAnalyzer(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # wget
        # https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav
        # -O records/asr_example_zh.wav
        cls.audio_file = os.path.join(TEST_DIR, "audio_files/asr_example_zh.wav")
        # Use an environment variable to get the ASR model TAG
        cls.tag = os.getenv("TURN_ANALYZER_TAG", "v2_smart_turn_analyzer")
        cls.model_name_or_path = os.getenv(
            "TURN_MODEL_PATH", os.path.join(MODELS_DIR, "pipecat-ai/smart-turn-v2")
        )
        cls.torch_dtype = os.getenv("TURN_TORCH_DTYPE", "float32")
        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    async def asyncSetUp(self):
        kwargs = {}
        kwargs["model_path"] = self.model_name_or_path
        kwargs["torch_dtype"] = self.torch_dtype

        self.turn: ITurnAnalyzer = TurnAnalyzerEnvInit.initTurnAnalyzerEngine(self.tag, kwargs)

        self.session = Session(**SessionCtx("test_client_id", 16000, 2).__dict__)

    async def asyncTearDown(self):
        pass

    async def test_end(self):
        assert self.turn.speech_triggered is False

        audio_bytes = b""
        with open(self.audio_file, "rb") as audio_file:
            audio_bytes = audio_file.read()
        state = self.turn.append_audio(audio_bytes, True)
        # print(state)
        assert self.turn.speech_triggered is True
        assert state == EndOfTurnState.INCOMPLETE

        state, result = await self.turn.analyze_end_of_turn()
        # print(state, result)
        assert self.turn.speech_triggered is False
        assert state == EndOfTurnState.COMPLETE
        assert result["prediction"] == 1

        self.turn.clear()
        assert self.turn.speech_triggered is False

    async def test_no_end(self):
        assert self.turn.speech_triggered is False

        audio = AudioSegment.from_file(self.audio_file, format="wav")

        total_duration = len(audio)
        duration_to_capture = int(total_duration * 3 / 4)
        audio_clip = audio[:duration_to_capture]

        tmp_path = os.path.join(RECORDS_DIR, "turn_temp_clip.wav")
        audio_clip.export(tmp_path, format="wav")
        print(f"Exported temp clip to {tmp_path}")
        assert os.path.exists(tmp_path)

        with wave.open(tmp_path, "rb") as wav_file:
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            audio_bytes = wav_file.readframes(n_frames)

            print(f"Channels: {n_channels}")
            print(f"Sample Width: {sample_width}")
            print(f"Frame Rate: {frame_rate}")
            print(f"Number of Frames: {n_frames}")
            print(f"Audio Data length: {len(audio_bytes)}")

            state = self.turn.append_audio(audio_bytes, True)
            # print(state)
            assert self.turn.speech_triggered is True
            assert state == EndOfTurnState.INCOMPLETE

            state, result = await self.turn.analyze_end_of_turn()
            print(state, result)
            assert self.turn.speech_triggered is True
            assert state == EndOfTurnState.INCOMPLETE
            assert result["prediction"] != 1

            self.turn.clear()
            assert self.turn.speech_triggered is False
