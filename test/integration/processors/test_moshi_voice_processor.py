import os
import logging

import unittest
from apipeline.pipeline.pipeline import Pipeline, FrameDirection
from apipeline.pipeline.task import PipelineTask, PipelineParams
from apipeline.pipeline.runner import PipelineRunner
from apipeline.processors.logger import FrameLogger
from apipeline.frames import TextFrame, AudioRawFrame

from src.processors.voice.moshi_voice_processor import MoshiVoiceProcessor
from src.modules.speech.vad_analyzer.silero import SileroVADAnalyzer
from src.common.types import DailyParams
from src.common.logger import Logger
from src.transports.daily import DailyTransport
from src.types.llm.lmgen import LMGenArgs

from dotenv import load_dotenv

load_dotenv(override=True)

"""
python -m unittest test.integration.processors.test_moshi_voice_processor.TestProcessor
"""


class TestProcessor(unittest.IsolatedAsyncioTestCase):

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
            audio_out_enabled=True,
        )
        if self.vad_enabled is True:
            daily_pramams.vad_enabled = True
            daily_pramams.vad_analyzer = SileroVADAnalyzer()
            daily_pramams.vad_audio_passthrough = True
        transport = DailyTransport(
            self.room_url,
            self.room_token,
            "daily moshi voice bot",
            daily_pramams,
        )

        voice_processor = MoshiVoiceProcessor(lm_gen_args=LMGenArgs())

        pipeline = Pipeline([
            transport.input_processor(),
            voice_processor,  # output TextFrame and AudioRawFrame
            FrameLogger(include_frame_types=[TextFrame, AudioRawFrame]),
            transport.output_processor(),
        ])

        self.task = PipelineTask(
            pipeline,
            PipelineParams(
                allow_interruptions=False,  # close pipeline interruptions, use model interrupt
                enable_metrics=True,
            )
        )

    async def asyncTearDown(self):
        pass

    async def test_asr(self):
        runner = PipelineRunner()
        await runner.run(self.task)
