import logging
from typing import Any, Optional

from apipeline.processors.frame_processor import FrameDirection

from src.processors.small_webrtc_input_transport_processor import SmallWebRTCInputProcessor
from src.processors.small_webrtc_output_transport_processor import SmallWebRTCOutputProcessor
from src.services.small_webrtc_client import SmallWebRTCCallbacks, SmallWebRTCClient
from src.common.types import AudioCameraParams
from src.services.webrtc_peer_connection import SmallWebRTCConnection
from src.transports.base import BaseTransport
from src.types.frames import SpriteFrame, OutputAudioRawFrame, OutputImageRawFrame


class SmallWebRTCTransport(BaseTransport):
    def __init__(
        self,
        webrtc_connection: SmallWebRTCConnection,
        params: AudioCameraParams,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        super().__init__(input_name=input_name, output_name=output_name)
        self._params = params

        self._callbacks = SmallWebRTCCallbacks(
            on_app_message=self._on_app_message,
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
        )

        self._client = SmallWebRTCClient(webrtc_connection, self._callbacks)

        self._input: Optional[SmallWebRTCInputProcessor] = None
        self._output: Optional[SmallWebRTCOutputProcessor] = None

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_app_message")
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")

    def input_processor(self) -> SmallWebRTCInputProcessor:
        if not self._input:
            self._input = SmallWebRTCInputProcessor(
                self._client, self._params, name=self._input_name
            )
        return self._input

    def output_processor(self) -> SmallWebRTCOutputProcessor:
        if not self._output:
            self._output = SmallWebRTCOutputProcessor(
                self._client, self._params, name=self._input_name
            )
        return self._output

    async def send_image(self, frame: OutputImageRawFrame | SpriteFrame):
        if self._output:
            await self._output.queue_frame(frame, FrameDirection.DOWNSTREAM)

    async def send_audio(self, frame: OutputAudioRawFrame):
        if self._output:
            await self._output.queue_frame(frame, FrameDirection.DOWNSTREAM)

    async def _on_app_message(self, webrtc_connection, message: Any):
        # if self._input:
        #    await self._input.push_app_message(message)
        await self._call_event_handler("on_app_message", webrtc_connection, message)

    async def _on_client_connected(self, webrtc_connection):
        await self._call_event_handler("on_client_connected", webrtc_connection)

    async def _on_client_disconnected(self, webrtc_connection):
        await self._call_event_handler("on_client_disconnected", webrtc_connection)
