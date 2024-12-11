import asyncio
import os
import logging

import unittest
from livekit import rtc
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.runner import PipelineRunner
from apipeline.pipeline.task import PipelineTask, PipelineParams
from apipeline.frames.control_frames import EndFrame
from apipeline.frames.data_frames import TextFrame

from src.services.help.livekit_room import LivekitRoom
from src.services.help import RoomManagerEnvInit
from src.common.types import LivekitParams
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
            ),
        )
        self.assertGreater(len(transport.event_names), 0)
        tts_processor = TTSProcessor(tts=self.tts)
        await tts_processor.set_voice("zh-CN-XiaoxiaoNeural")

        task = PipelineTask(
            Pipeline([tts_processor, transport.output_processor()]),
            params=PipelineParams(allow_interruptions=True),
        )

        # Register an event handler so we can play the audio
        # when the participant joins.
        # @transport.event_handler("on_participant_connected")
        @transport.event_handler("on_first_participant_joined")
        async def on_new_participant_joined(
            transport: LivekitTransport, participant: rtc.RemoteParticipant
        ):
            print("transport---->", transport)
            print("participant---->", participant)
            participant_name = participant.name if participant.name else participant.identity
            await transport.send_message(
                f"hello,你好，{participant_name}, 我是机器人。",
                participant_id=participant.identity,
            )
            await task.queue_frames(
                [
                    TextFrame(f"你好，Hello there. {participant_name},"),
                    TextFrame("你是一个中国人。"),
                    EndFrame(),
                ]
            )

        runner = PipelineRunner()
        await runner.run(task)
