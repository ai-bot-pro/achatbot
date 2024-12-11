import asyncio
import os
import logging

import unittest
from agora_realtime_ai_api import rtc
from agora_realtime_ai_api.token_builder import realtimekit_token_builder
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.runner import PipelineRunner
from apipeline.pipeline.task import PipelineTask, PipelineParams
from apipeline.frames.control_frames import EndFrame
from apipeline.processors.logger import FrameLogger
from apipeline.frames.data_frames import TextFrame, DataFrame, AudioRawFrame, ImageRawFrame

from src.services.help.agora.channel import AgoraChannel
from src.services.help import RoomManagerEnvInit
from src.common.types import AgoraParams
from src.common.logger import Logger
from src.transports.agora import AgoraTransport

from dotenv import load_dotenv

load_dotenv(override=True)

r"""
ROOM_NAME=bot-room \
    python -m unittest test.integration.processors.test_agora_echo_processor.TestProcessor

ROOM_NAME=bot-room RUN_TIMEOUT=3600 \
    python -m unittest test.integration.processors.test_agora_echo_processor.TestProcessor
"""


class TestProcessor(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        logger = Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)
        cls.room_name = os.getenv("ROOM_NAME", "chat-room")
        rtc.logger = logger
        realtimekit_token_builder.logger = logger

    @classmethod
    def tearDownClass(cls):
        pass

    async def asyncSetUp(self):
        self.agora_rtc_channel: AgoraChannel = RoomManagerEnvInit.initEngine("agora_rtc_channel")

    async def asyncTearDown(self):
        await self.agora_rtc_channel.close_session()

    async def test_echo(self):
        token = await self.agora_rtc_channel.gen_token(self.room_name)
        self.params = AgoraParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16000,
            audio_in_channels=1,
            audio_out_enabled=True,
            audio_out_sample_rate=16000,
            audio_out_channels=1,
            camera_in_enabled=True,
            camera_out_enabled=True,
            camera_out_width=640,  # from Video Profiles https://webdemo.agora.io/basicVideoCall/index.html
            camera_out_height=480,
            camera_out_framerate=30,
        )
        transport = AgoraTransport(token, self.params)
        self.assertGreater(len(transport.event_names), 0)

        self.task = PipelineTask(
            Pipeline(
                [
                    transport.input_processor(),
                    # FrameLogger(include_frame_types=[AudioRawFrame]),
                    # FrameLogger(include_frame_types=[ImageRawFrame]),
                    transport.output_processor(),
                ]
            ),
            params=PipelineParams(allow_interruptions=False),
        )

        transport.add_event_handler("on_connected", self.on_connected)
        transport.add_event_handler("on_connection_state_changed", self.on_connection_state_changed)
        transport.add_event_handler("on_connection_failure", self.on_connection_failure)
        transport.add_event_handler("on_disconnected", self.on_disconnected)
        transport.add_event_handler("on_error", self.on_error)
        transport.add_event_handler("on_first_participant_joined", self.on_first_participant_joined)
        transport.add_event_handler("on_participant_disconnected", self.on_participant_disconnected)
        # sometime don't get on_connection_state_changed when disconnected
        transport.add_event_handler("on_connection_state_changed", self.on_connection_state_changed)
        transport.add_event_handler(
            "on_audio_subscribe_state_changed", self.on_audio_subscribe_state_changed
        )
        transport.add_event_handler("on_data_received", self.on_data_received)
        transport.add_event_handler(
            "on_video_subscribe_state_changed", self.on_video_subscribe_state_changed
        )

        runner = PipelineRunner()
        try:
            await asyncio.wait_for(runner.run(self.task), int(os.getenv("RUN_TIMEOUT", "60")))
        except asyncio.TimeoutError:
            logging.warning("Test run timeout. Exiting")
            await self.task.queue_frame(EndFrame())

    async def on_connected(
        self,
        transport: AgoraTransport,
        agora_rtc_conn: rtc.RTCConnection,
        conn_info: rtc.RTCConnInfo,
        reason: int,
    ):
        logging.info(f"agora_rtc_conn:{agora_rtc_conn} conn_info:{conn_info} reason:{reason}")

    async def on_connection_failure(self, transport: AgoraTransport, reason: int):
        logging.info(f"reason:{reason}")

    async def on_error(self, transport: AgoraTransport, error_msg: str):
        logging.info(f"error_msg:{error_msg}")

    async def on_first_participant_joined(self, transport: AgoraTransport, user_id: str):
        logging.info(f"user_id:{user_id}")

        if self.params.audio_in_enabled:
            transport.capture_participant_audio(user_id)
        if self.params.camera_in_enabled:
            # passive capture one image
            # transport.capture_participant_video(user_id, framerate=0)
            # active to capture
            transport.capture_participant_video(user_id)

        await transport.send_message(
            "hello,你好，我是机器人。",
            participant_id=user_id,
        )

    async def on_participant_disconnected(
        self,
        transport: AgoraTransport,
        agora_rtc_conn: rtc.RTCConnection,
        user_id: str,
        reason: int,
    ):
        logging.info(f"Partcipant {user_id} left. reason:{reason}")
        logging.info(f"current remote Partcipants {transport.get_participant_ids()}")

    async def on_disconnected(
        self,
        transport: AgoraTransport,
        agora_rtc_conn: rtc.RTCConnection,
        conn_info: rtc.RTCConnInfo,
        reason: int,
    ):
        logging.info("disconnected reason %s, Exiting." % reason)
        await self.task.queue_frame(EndFrame())

    async def on_connection_state_changed(
        self,
        transport: AgoraTransport,
        agora_rtc_conn: rtc.RTCConnection,
        conn_info: rtc.RTCConnInfo,
        reason: int,
    ):
        logging.info(f"connection state {conn_info.state} reason:{reason}")

    async def on_audio_subscribe_state_changed(
        self,
        transport: AgoraTransport,
        agora_local_user: rtc.LocalUser,
        channel: str,
        user_id: str,
        old_state: int,
        new_state: int,
        elapse_since_last_state: int,
    ):
        logging.info(
            f"agora_local_user:{agora_local_user}, channel:{channel}, user_id:{user_id}, old_state:{old_state}, new_state:{new_state}, elapse_since_last_state:{elapse_since_last_state}"
        )

    async def on_data_received(self, transport: AgoraTransport, data: bytes, user_id: str):
        logging.info(f"len(data):{len(data)} user_id:{user_id}")

    async def on_video_subscribe_state_changed(
        self,
        transport: AgoraTransport,
        agora_local_user: rtc.LocalUser,
        channel: str,
        user_id: str,
        old_state: int,
        new_state: int,
        elapse_since_last_state: int,
    ):
        logging.info(
            f"agora_local_user:{agora_local_user}, channel:{channel}, user_id:{user_id}, old_state:{old_state}, new_state:{new_state}, elapse_since_last_state:{elapse_since_last_state}"
        )
