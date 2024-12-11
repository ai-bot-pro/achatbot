import asyncio
import os
from typing import List, Optional
import logging

from agora_realtime_ai_api import rtc
from apipeline.frames.control_frames import EndFrame
from apipeline.processors.frame_processor import FrameDirection
from apipeline.frames.data_frames import AudioRawFrame

from src.processors.agora_input_transport_processor import AgoraInputTransportProcessor
from src.processors.agora_output_transport_processor import AgoraOutputTransportProcessor
from src.common.types import AgoraParams
from src.services.agora_client import AgoraCallbacks, AgoraTransportClient
from src.transports.base import BaseTransport
from src.types.frames.data_frames import AgoraTransportMessageFrame


class AgoraTransport(BaseTransport):
    def __init__(
        self,
        token: str,
        params: AgoraParams = AgoraParams(),
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        super().__init__(input_name=input_name, output_name=output_name, loop=loop)

        self._register_event_handler("on_connected")
        self._register_event_handler("on_connection_state_changed")
        self._register_event_handler("on_connection_failure")
        self._register_event_handler("on_disconnected")
        self._register_event_handler("on_error")
        self._register_event_handler("on_participant_connected")
        self._register_event_handler("on_participant_disconnected")
        self._register_event_handler("on_data_received")
        self._register_event_handler("on_first_participant_joined")
        self._register_event_handler("on_audio_subscribe_state_changed")
        self._register_event_handler("on_video_subscribe_state_changed")
        logging.info(f"AgoraTransport register event names: {self.event_names}")
        callbacks = AgoraCallbacks(
            on_connected=self._on_connected,
            on_connection_state_changed=self._on_connection_state_changed,
            on_connection_failure=self._on_connection_failure,
            on_disconnected=self._on_disconnected,
            on_error=self._on_error,
            on_participant_connected=self._on_participant_connected,
            on_participant_disconnected=self._on_participant_disconnected,
            on_data_received=self._on_data_received,
            on_first_participant_joined=self._on_first_participant_joined,
            on_audio_subscribe_state_changed=self._on_audio_subscribe_state_changed,
            on_video_subscribe_state_changed=self._on_video_subscribe_state_changed,
        )

        self._params = params
        self._params.app_id = params.app_id or os.getenv("AGORA_APP_ID")
        self._params.app_cert = params.app_cert or os.getenv("AGORA_APP_CERT")
        self._client = AgoraTransportClient(
            token,
            self._params,
            callbacks,
            self._loop,
        )
        logging.debug(f"AgoraTransport params:{self._params}")

        self._input: AgoraInputTransportProcessor | None = None
        self._output: AgoraOutputTransportProcessor | None = None

    def input_processor(self) -> AgoraInputTransportProcessor:
        if not self._input:
            self._input = AgoraInputTransportProcessor(
                self._client, self._params, name=self._input_name
            )
        return self._input

    def output_processor(self) -> AgoraOutputTransportProcessor:
        if not self._output:
            self._output = AgoraOutputTransportProcessor(
                self._client, self._params, name=self._output_name
            )
        return self._output

    async def send_audio(self, frame: AudioRawFrame):
        if self._output:
            await self._output.process_frame(frame, FrameDirection.DOWNSTREAM)

    def get_participant_ids(self) -> List[int]:
        return self._client.get_participant_ids()

    async def get_participant_metadata(self, participant_id: str) -> dict:
        return await self._client.get_participant_metadata(participant_id)

    async def _on_connected(
        self, agora_rtc_conn: rtc.RTCConnection, conn_info: rtc.RTCConnInfo, reason: str
    ):
        await self._call_event_handler("on_connected", agora_rtc_conn, conn_info, reason)

    async def _on_first_participant_joined(self, user_id: int):
        await self._call_event_handler("on_first_participant_joined", user_id)

    async def _on_error(self, error_msg: str):
        await self._call_event_handler("on_error", error_msg)

    async def _on_connection_state_changed(
        self, agora_rtc_conn: rtc.RTCConnection, conn_info: rtc.RTCConnInfo, reason: str
    ):
        await self._call_event_handler(
            "on_connection_state_changed", agora_rtc_conn, conn_info, reason
        )

    async def _on_connection_failure(self, error_msg: str):
        await self._call_event_handler("on_connection_failure", error_msg)

    async def _on_disconnected(
        self, agora_rtc_conn: rtc.RTCConnection, conn_info: rtc.RTCConnInfo, reason: str
    ):
        await self._call_event_handler("on_disconnected", agora_rtc_conn, conn_info, reason)

    async def _on_participant_connected(self, agora_rtc_conn: rtc.RTCConnection, user_id: int):
        await self._call_event_handler("on_participant_connected", agora_rtc_conn, user_id)

    async def _on_participant_disconnected(
        self, agora_rtc_conn: rtc.RTCConnection, user_id: int, reason: str
    ):
        await self._call_event_handler(
            "on_participant_disconnected", agora_rtc_conn, user_id, reason
        )
        if self._input:
            await self._input.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)
        if self._output:
            await self._output.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)

    async def _on_data_received(self, data: bytes, user_id: int):
        if self._input:
            await self._input.push_app_message(data.decode(), str(user_id))
        await self._call_event_handler("on_data_received", data, user_id)

    async def _on_audio_subscribe_state_changed(
        self,
        agora_local_user: rtc.LocalUser,
        channel: rtc.Channel,
        user_id: int,
        old_state: int,
        new_state: int,
        elapse_since_last_state: int,
    ):
        await self._call_event_handler(
            "on_audio_subscribe_state_changed",
            agora_local_user,
            channel,
            user_id,
            old_state,
            new_state,
            elapse_since_last_state,
        )

    async def _on_video_subscribe_state_changed(
        self,
        agora_local_user: rtc.LocalUser,
        channel: rtc.Channel,
        user_id: int,
        old_state: int,
        new_state: int,
        elapse_since_last_state: int,
    ):
        await self._call_event_handler(
            "on_video_subscribe_state_changed",
            agora_local_user,
            channel,
            user_id,
            old_state,
            new_state,
            elapse_since_last_state,
        )

    async def send_message(self, message: str, participant_id: str | None = None):
        if self._output:
            frame = AgoraTransportMessageFrame(message=message, participant_id=participant_id)
            await self._output.send_message(frame)

    async def cleanup(self):
        if self._input:
            await self._input.cleanup()
        if self._output:
            await self._output.cleanup()
        await self._client.leave()

    def capture_participant_audio(
        self,
        participant_id: str,
        *,
        sample_rate=None,
        num_channels=None,
    ):
        if self._input:
            self._input.capture_participant_audio(participant_id, sample_rate, num_channels)

    def capture_participant_video(
        self,
        participant_id: str,
        *,
        framerate: int = 30,
        video_source: str = "camera",
        color_format: str = "RGB",
    ):
        r"""
        ### !NOTE: need AgoraParams camera_in_enabled=True
        - if room just a bot, then participant join room
         do capture_participant_video
         after agora subscribed a participant video
        - if want switch the latest subscribed participant
        """
        if self._input:
            self._input.capture_participant_video(
                participant_id, framerate, video_source, color_format
            )
