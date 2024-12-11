import logging
import os

from agora_realtime_ai_api import rtc
from agora_realtime_ai_api.token_builder import realtimekit_token_builder
from apipeline.frames.control_frames import EndFrame

from src.common.logger import Logger
from src.cmd.bots.base import AIChannelBot
from src.transports.agora import AgoraTransport

# Init logger, use monkey fix to replace agora_realtime_ai_api logger  :)
logger = Logger.logger or Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)
rtc.logger = logger
realtimekit_token_builder.logger = logger


class AgoraChannelBot(AIChannelBot):
    def regisiter_room_event(self, transport: AgoraTransport):
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

    async def on_connected(
        self,
        transport: AgoraTransport,
        agora_rtc_conn: rtc.RTCConnection,
        conn_info: rtc.RTCConnInfo,
        reason: int,
    ):
        logging.debug(f"agora_rtc_conn:{agora_rtc_conn} conn_info:{conn_info} reason:{reason}")

    async def on_connection_failure(self, transport: AgoraTransport, reason: int):
        logging.debug(f"reason:{reason}")

    async def on_error(self, transport: AgoraTransport, error_msg: str):
        logging.error(f"error_msg:{error_msg}")

    async def on_first_participant_joined(self, transport: AgoraTransport, user_id: str):
        logging.info(f"fisrt joined user_id:{user_id}")

        # TODO: need know room anchor participant
        # now just support one by one chat with bot
        # transport.capture_participant_audio(
        #    participant_id=user_id,
        # )

    async def on_participant_disconnected(
        self,
        transport: AgoraTransport,
        agora_rtc_conn: rtc.RTCConnection,
        user_id: str,
        reason: int,
    ):
        logging.info(f"Partcipant {user_id} left. reason:{reason}")

    async def on_disconnected(
        self,
        transport: AgoraTransport,
        agora_rtc_conn: rtc.RTCConnection,
        conn_info: rtc.RTCConnInfo,
        reason: int,
    ):
        logging.info(f"disconnected reason {reason}, Exiting.")
        await self.task.queue_frame(EndFrame())

    async def on_connection_state_changed(
        self,
        transport: AgoraTransport,
        agora_rtc_conn: rtc.RTCConnection,
        conn_info: rtc.RTCConnInfo,
        reason: int,
    ):
        logging.info(f"connection state {conn_info.state} reason:{reason}")
        if conn_info.state == 5:
            await self.task.queue_frame(EndFrame())

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
        logging.info(f"size:{len(data)} from user_id:{user_id}")

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
