import os
import logging

import unittest
from apipeline.pipeline.pipeline import Pipeline, FrameDirection
from apipeline.pipeline.task import PipelineTask, PipelineParams
from apipeline.pipeline.runner import PipelineRunner
from apipeline.frames.data_frames import DataFrame, TextFrame, Frame
from apipeline.frames.control_frames import EndFrame
from apipeline.processors.frame_processor import FrameProcessor

from src.modules.speech.vad_analyzer.silero import SileroVADAnalyzer
from src.processors.speech.asr.deepgram_asr_processor import DeepgramAsrProcessor
from src.common.types import DailyParams
from src.common.logger import Logger
from src.transports.daily import DailyTransport
from src.types.frames.data_frames import TranscriptionFrame
from src.common.session import Session
from src.common.types import SessionCtx

from dotenv import load_dotenv

load_dotenv(override=True)

"""
python -m unittest test.integration.processors.test_asr_deepgram_processor.TestASRDeepgramProcessor
DEEPGRAM_LANGUAGE=zh \
    python -m unittest test.integration.processors.test_asr_deepgram_processor.TestASRDeepgramProcessor
"""


class TranscriptionLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            print(f"get Transcription Frame: {frame}, text len:{len(frame.text)}")


class TestASRDeepgramProcessor(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)
        cls.room_url = os.getenv("DAILY_ROOM_URL", "https://weedge.daily.co/chat-room")
        cls.room_token = os.getenv("DAILY_ROOM_TOKEN", None)
        cls.vad_enabled = bool(os.getenv("DAILY_VAD_ENABLED", ""))

    @classmethod
    def tearDownClass(cls):
        pass

    async def asyncSetUp(self):
        daily_pramams: DailyParams = DailyParams(
            audio_in_enabled=True,
            transcription_enabled=False,
        )
        if self.vad_enabled is True:
            daily_pramams.vad_enabled = True
            daily_pramams.vad_analyzer = SileroVADAnalyzer()
            daily_pramams.vad_audio_passthrough = True
        transport = DailyTransport(
            self.room_url,
            self.room_token,
            "Transcription bot",
            daily_pramams,
        )

        asr_processor = DeepgramAsrProcessor(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            language=os.getenv("DEEPGRAM_LANGUAGE", "en"),
        )

        tl_porcessor = TranscriptionLogger()

        pipeline = Pipeline([transport.input_processor(), asr_processor, tl_porcessor])

        self.task = PipelineTask(
            pipeline,
            PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
            ),
        )

    async def asyncTearDown(self):
        pass

    async def test_asr(self):
        runner = PipelineRunner()
        await runner.run(self.task)
