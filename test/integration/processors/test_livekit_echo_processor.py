import asyncio
import os
import logging

import unittest
from livekit import rtc
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.runner import PipelineRunner
from apipeline.pipeline.task import PipelineTask, PipelineParams
from apipeline.frames.control_frames import EndFrame
from apipeline.processors.logger import FrameLogger
from apipeline.frames.data_frames import TextFrame, DataFrame, AudioRawFrame, ImageRawFrame

from src.services.help.livekit_room import LivekitRoom
from src.services.help import RoomManagerEnvInit
from src.common.types import LivekitParams
from src.common.logger import Logger
from src.transports.livekit import LivekitTransport

from dotenv import load_dotenv

load_dotenv(override=True)

r"""
LIVEKIT_BOT_NAME=chat-bot ROOM_NAME=bot-room \
    python -m unittest test.integration.processors.test_livekit_echo_processor.TestProcessor

LIVEKIT_BOT_NAME=chat-bot ROOM_NAME=bot-room RUN_TIMEOUT=3600 \
    python -m unittest test.integration.processors.test_livekit_echo_processor.TestProcessor
"""


class TestProcessor(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)
        cls.room_name = os.getenv("ROOM_NAME", "chat-room")

    @classmethod
    def tearDownClass(cls):
        pass

    async def asyncSetUp(self):
        self.livekit_room: LivekitRoom = RoomManagerEnvInit.initEngine("livekit_room")

    async def asyncTearDown(self):
        await self.livekit_room.close_session()

    async def test_echo(self):
        token = await self.livekit_room.gen_token(self.room_name)
        self.params = LivekitParams(
            audio_in_enabled=True,
            # audio_in_participant_enabled=True,
            audio_in_sample_rate=16000,
            audio_in_channels=1,
            audio_out_enabled=True,
            audio_out_sample_rate=16000,
            audio_out_channels=1,
        )
        transport = LivekitTransport(token, self.params)
        self.assertGreater(len(transport.event_names), 0)

        self.task = PipelineTask(
            Pipeline(
                [
                    transport.input_processor(),
                    # FrameLogger(include_frame_types=[AudioRawFrame]),
                    transport.output_processor(),
                ]
            ),
            params=PipelineParams(allow_interruptions=True),
        )

        transport.add_event_handler("on_connected", self.on_connected)
        transport.add_event_handler("on_error", self.on_error)
        transport.add_event_handler("on_first_participant_joined", self.on_first_participant_joined)
        transport.add_event_handler("on_participant_disconnected", self.on_participant_disconnected)
        transport.add_event_handler("on_disconnected", self.on_disconnected)
        # sometime don't get on_connection_state_changed when disconnected
        transport.add_event_handler("on_connection_state_changed", self.on_connection_state_changed)

        transport.add_event_handler("on_audio_track_subscribed", self.on_audio_track_subscribed)
        transport.add_event_handler("on_audio_track_unsubscribed", self.on_audio_track_unsubscribed)
        transport.add_event_handler("on_video_track_subscribed", self.on_video_track_subscribed)
        transport.add_event_handler("on_video_track_unsubscribed", self.on_video_track_unsubscribed)

        runner = PipelineRunner()
        try:
            await asyncio.wait_for(runner.run(self.task), int(os.getenv("RUN_TIMEOUT", "60")))
        except asyncio.TimeoutError:
            logging.warning("Test run timeout. Exiting")
            await self.task.queue_frame(EndFrame())

    async def on_connected(self, transport: LivekitTransport, room: rtc.Room):
        print("room--->", room)

    async def on_error(self, transport: LivekitTransport, error_msg: str):
        print("error_msg--->", error_msg)

    async def on_first_participant_joined(
        self, transport: LivekitTransport, participant: rtc.RemoteParticipant
    ):
        print(f"on_first_participant_joined---->{participant}")

        # TODO: need know room anchor participant
        # now just support one by one chat with bot
        # need open audio_in_participant_enabled
        transport.capture_participant_audio(
            participant_id=participant.sid,
        )

        participant_name = participant.name if participant.name else participant.identity
        await transport.send_message(
            f"hello,你好，{participant_name}, 我是机器人。",
            participant_id=participant.identity,
        )

    async def on_participant_disconnected(
        self, transport: LivekitTransport, participant: rtc.RemoteParticipant
    ):
        logging.info(f"Partcipant {participant} left.")
        logging.info(f"current remote Partcipants {transport.get_participants()}")

    async def on_disconnected(self, transport: LivekitTransport, reason: str):
        logging.info("disconnected reason %s, Exiting." % reason)
        await self.task.queue_frame(EndFrame())

    async def on_connection_state_changed(
        self, transport: LivekitTransport, state: rtc.ConnectionState
    ):
        logging.info("connection state %s " % state)
        if state == rtc.ConnectionState.CONN_DISCONNECTED:
            await self.task.queue_frame(EndFrame())

    async def on_audio_track_subscribed(
        self, transport: LivekitTransport, participant: rtc.RemoteParticipant
    ):
        print(f"on_audio_track_subscribed---->{participant}")

    async def on_audio_track_unsubscribed(
        self, transport: LivekitTransport, participant: rtc.RemoteParticipant
    ):
        print(f"on_audio_track_unsubscribed---->{participant}")

    async def on_video_track_subscribed(
        self, transport: LivekitTransport, participant: rtc.RemoteParticipant
    ):
        print(f"on_video_track_subscribed---->{participant}")

    async def on_video_track_unsubscribed(
        self, transport: LivekitTransport, participant: rtc.RemoteParticipant
    ):
        print(f"on_video_track_unsubscribed---->{participant}")
