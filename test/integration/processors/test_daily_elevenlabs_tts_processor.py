import asyncio
import os
import logging

import unittest
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.runner import PipelineRunner
from apipeline.pipeline.task import PipelineTask, PipelineParams
from apipeline.frames.control_frames import EndFrame
from apipeline.frames.data_frames import TextFrame

from src.common.types import DailyParams
from src.services.help.daily_rest import DailyRESTHelper
from src.common.logger import Logger
from src.processors.speech.tts.elevenlabs_tts_processor import ElevenLabsTTSProcessor
from src.transports.daily import DailyTransport

from dotenv import load_dotenv

load_dotenv(override=True)

r"""
DAILY_ROOM_URL=https://weedge.daily.co/chat-bot \
    python -m unittest test.integration.processors.test_daily_elevenlabs_tts_processor.TestElevenlabsTTSProcessor
"""


DAILY_API_URL = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")
DAILY_API_KEY = os.getenv("DAILY_API_KEY", "")
ROOM_EXPIRE_TIME = 30 * 60  # 30 minutes
ROOM_TOKEN_EXPIRE_TIME = 30 * 60  # 30 minutes


class TestElevenlabsTTSProcessor(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)
        # https://play.cartesia.ai/
        cls.voice_id = os.getenv("ELEVENLABS_VOICE_ID", "VGcvPRjFP4qKhICQHO7d")
        cls.room_url = os.getenv("DAILY_ROOM_URL", "https://weedge.daily.co/chat-bot")

    @classmethod
    def tearDownClass(cls):
        pass

    async def asyncSetUp(self):
        self.daily_rest_helper = DailyRESTHelper(DAILY_API_KEY, DAILY_API_URL)
        pass

    async def asyncTearDown(self):
        pass

    async def test_tts_daily_output(self):
        bot_name = "my test bot"
        transport = DailyTransport(
            self.room_url, None, bot_name, DailyParams(audio_out_enabled=True)
        )

        tts = ElevenLabsTTSProcessor(
            voice_id=self.voice_id,
            # sync_order_send=True,  # for send EndFrame in the end to test
        )

        task = PipelineTask(
            Pipeline([tts, transport.output_processor()]),
            params=PipelineParams(allow_interruptions=True),
        )

        # Register an event handler so we can play the audio when the
        # participant joins.
        @transport.event_handler("on_participant_joined")
        async def on_new_participant_joined(transport, participant):
            participant_name = bot_name
            if "userName" in participant["info"]:
                participant_name = participant["info"]["userName"]
            await task.queue_frames(
                [
                    TextFrame(f"你好，Hello there. {participant_name}"),
                    TextFrame("你是一个中国人。"),
                    TextFrame("一名中文助理，请用中文简短回答，回答限制在5句话内。"),
                    EndFrame(),
                ]
            )

        runner = PipelineRunner()
        await runner.run(task)
