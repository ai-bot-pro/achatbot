import logging
from typing import Any

from fastapi import WebSocket
from apipeline.frames.control_frames import EndFrame

from src.services.webrtc_peer_connection import SmallWebRTCConnection
from src.cmd.bots.base import AIBot
from src.transports.small_webrtc import SmallWebRTCTransport


class SmallWebrtcAIBot(AIBot):
    def __init__(self, webrtc_connection: SmallWebRTCConnection | None = None, **args) -> None:
        super().__init__(**args)
        self._webrtc_connection = webrtc_connection

    def set_webrtc_connection(self, webrtc_connection: SmallWebRTCConnection):
        self._webrtc_connection = webrtc_connection

    def register_event(self, transport: SmallWebRTCTransport):
        transport.add_event_handler(
            "on_client_connected",
            self.on_client_connected,
        )
        transport.add_event_handler(
            "on_client_disconnected",
            self.on_client_disconnected,
        )
        transport.add_event_handler(
            "on_app_message",
            self.on_app_message,
        )

    async def on_client_connected(
        self,
        transport: SmallWebRTCTransport,
        connection: SmallWebRTCConnection,
    ):
        logging.info(f"on_client_connected callee id:{connection.pc_id}")
        self.session.set_client_id(connection.pc_id)

    async def on_client_disconnected(
        self,
        transport: SmallWebRTCTransport,
        connection: SmallWebRTCConnection,
    ):
        logging.info(f"on_client_disconnected callee id:{connection.pc_id}")
        if self.task is not None:
            await self.task.queue_frame(EndFrame())

    async def on_app_message(
        self,
        transport: SmallWebRTCTransport,
        message: Any,
    ):
        logging.info(f"on_app_message received message: {message}")
