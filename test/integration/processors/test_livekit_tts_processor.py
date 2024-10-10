import asyncio
import os
import logging

import unittest
from livekit import rtc, api
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.runner import PipelineRunner
from apipeline.pipeline.task import PipelineTask, PipelineParams
from apipeline.frames.control_frames import EndFrame
from apipeline.frames.data_frames import TextFrame

from src.services.help.livekit_room import LivekitRoom
from src.services.help import RoomManagerEnvInit
from src.common.types import DailyParams, LivekitParams
from src.common.logger import Logger
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.modules.speech.tts import TTSEnvInit
from src.transports.livekit import LivekitTransport

from dotenv import load_dotenv

load_dotenv(override=True)

r"""
LIVEKIT_BOT_NAME=chat-bot \
    python -m unittest test.integration.processors.test_livekit_tts_processor.TestTTSProcessor
"""


ROOM_EXPIRE_TIME = 30 * 60  # 30 minutes
ROOM_TOKEN_EXPIRE_TIME = 30 * 60  # 30 minutes


class TestTTSProcessor(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)
        cls.room_name = os.getenv("ROOM_NAME", "chat-room")
        cls.room_sandbox_url = os.getenv(
            "ROOM_SANDBOX_URL",
            f"https://ultra-terminal-re8nmd.sandbox.livekit.io/rooms/{cls.room_name}")

    @classmethod
    def tearDownClass(cls):
        pass

    async def asyncSetUp(self):
        self.livekit_room: LivekitRoom = RoomManagerEnvInit.initEngine("livekit_room")
        self.tts = TTSEnvInit.initTTSEngine()
        self.stream_info = self.tts.get_stream_info()

    async def asyncTearDown(self):
        await self.livekit_room.close_session()

    async def test_tts_daily_output(self):
        token = await self.livekit_room.gen_token(self.room_name)
        transport = LivekitTransport(
            token,
            params=LivekitParams(
                audio_out_enabled=True,
                audio_out_sample_rate=self.stream_info["rate"],
            )
        )
        self.assertGreater(len(transport.event_names), 0)
        tts_processor = TTSProcessor(tts=self.tts)
        await tts_processor.set_voice("zh-CN-XiaoxiaoNeural")

        task = PipelineTask(
            Pipeline([
                tts_processor,
                transport.output_processor()
            ]),
            params=PipelineParams(allow_interruptions=True)
        )

        print(
            f"bot {self.livekit_room.args.bot_name} joined room sandbox url",
            self.room_sandbox_url)

        # Register an event handler so we can play the audio
        # when the participant joins.
        @transport.event_handler("on_participant_connected")
        @transport.event_handler("on_first_participant_joined")
        async def on_new_participant_joined(
                transport: LivekitTransport,
                participant: rtc.RemoteParticipant):
            print("transport", transport)
            print("participant", participant)
            participant_name = participant.name if participant.name else participant.identity
            await task.queue_frames([
                TextFrame(f"你好，Hello there. {participant_name},"),
                TextFrame(f"你是一个中国人。"),
                # TextFrame(f"一名中文助理，请用中文简短回答，回答限制在5句话内。"),
                # TextFrame(f"我是Andrej Karpathy，我在YouTube上发布关于机器学习和深度学习的视频。如果你有任何关于这些视频的疑问或需要帮助，请告诉我！"),
                EndFrame(),
            ])

        runner = PipelineRunner()
        await runner.run(task)
