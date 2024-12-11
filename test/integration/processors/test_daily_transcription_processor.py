import os
import logging

import unittest
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineTask, PipelineParams
from apipeline.pipeline.runner import PipelineRunner
from apipeline.frames.data_frames import DataFrame, TextFrame
from apipeline.frames.control_frames import EndFrame
from apipeline.processors.output_processor import OutputFrameProcessor

from src.modules.speech.vad_analyzer.silero import SileroVADAnalyzer
from src.common.types import DailyParams, DailyTranscriptionSettings
from src.common.logger import Logger
from src.transports.daily import DailyTransport

from dotenv import load_dotenv

load_dotenv(override=True)

r"""
DAILY_TRANSCRIPTION_LANG=en \
DAILY_ROOM_URL=https://weedge.daily.co/chat-bot \
    DAILY_ROOM_TOKEN=${TOKEN} \
    python -m unittest test.integration.processors.test_daily_transcription_processor

DAILY_TRANSCRIPTION_LANG=zh \
    DAILY_ROOM_URL=https://weedge.daily.co/chat-bot \
    DAILY_ROOM_TOKEN=${TOKEN} \
    python -m unittest test.integration.processors.test_daily_transcription_processor
"""


class TestDailyTranscriptionProcessor(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)
        cls.room_url = os.getenv("DAILY_ROOM_URL", "https://weedge.daily.co/chat-bot")
        cls.room_token = os.getenv("DAILY_ROOM_TOKEN", None)
        cls.language = os.getenv("DAILY_TRANSCRIPTION_LANG", "en")

    @classmethod
    def tearDownClass(cls):
        pass

    async def asyncSetUp(self):
        bot_name = "my test bot"
        if self.language == "en":
            transcription_settings = DailyTranscriptionSettings(
                language=self.language,
            )
        if self.language == "zh":
            transcription_settings = DailyTranscriptionSettings(
                language=self.language,
                # tier="nova",
                # model="nova-2",
            )

        transport = DailyTransport(
            self.room_url,
            self.room_token,
            bot_name,
            DailyParams(
                audio_in_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                transcription_enabled=True,
                transcription_settings=transcription_settings,
            ),
        )

        self.texts = []
        out_processor = OutputFrameProcessor(cb=self.sink_callback)

        pipeline = Pipeline([transport.input_processor(), out_processor])
        self.task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

        transport.add_event_handler("on_first_participant_joined", self.on_first_participant_joined)
        transport.add_event_handler("on_participant_left", self.on_participant_left)
        transport.add_event_handler("on_call_state_updated", self.on_call_state_updated)

    async def asyncTearDown(self):
        pass

    async def sink_callback(self, frame: DataFrame):
        print(f"sink_callback ----> print frame: {frame}")
        if not isinstance(frame, TextFrame):
            return

        self.texts.append(frame.text)
        if len(self.texts) == 2:  # asr 2 times (speech to text)
            print("sink_callback ----> send end frame")
            await self.task.queue_frame(EndFrame())
            return

    async def on_first_participant_joined(self, transport: DailyTransport, participant):
        # register transcription event handler, exec it when daily event is received
        transport.capture_participant_transcription(participant["id"])
        logging.info("First participant joined")

    async def on_participant_left(self, transport: DailyTransport, participant, reason):
        await self.task.queue_frame(EndFrame())
        logging.info("Partcipant left. Exiting.")

    async def on_call_state_updated(self, transport: DailyTransport, state):
        logging.info("Call state %s " % state)
        if state == "left":
            await self.task.queue_frame(EndFrame())

    async def test_transcription(self):
        runner = PipelineRunner()
        await runner.run(self.task)
        self.assertGreater(len(self.texts), 0)
