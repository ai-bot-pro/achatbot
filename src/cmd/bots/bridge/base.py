import logging
from typing import Any

from fastapi import WebSocket
from apipeline.frames.control_frames import EndFrame

from src.cmd.bots.base import AIBot
from src.services.webrtc_peer_connection import SmallWebRTCConnection
from src.transports.fastapi_websocket_server import FastapiWebsocketTransport
from src.transports.small_webrtc import SmallWebRTCTransport
from src.types.frames import TransportMessageFrame


class AISmallWebRTCFastapiWebsocketBot(AIBot):
    def __init__(
        self,
        webrtc_connection: SmallWebRTCConnection | None = None,
        websocket: WebSocket | None = None,
        **args,
    ) -> None:
        super().__init__(**args)
        self.init_bot_config()
        self._webrtc_connection = webrtc_connection
        self._websocket = websocket

    def set_fastapi_websocket(self, websocket: WebSocket):
        self._websocket = websocket

    def set_webrtc_connection(self, webrtc_connection: SmallWebRTCConnection):
        self._webrtc_connection = webrtc_connection

    async def on_ws_client_connected(
        self,
        transport: FastapiWebsocketTransport,
        websocket: WebSocket,
    ):
        logging.info(f"on_ws_client_connected client:{websocket.client}")
        self.session.set_client_id(client_id=f"{websocket.client.host}:{websocket.client.port}")

    async def on_ws_client_disconnected(
        self,
        transport: FastapiWebsocketTransport,
        websocket: WebSocket,
    ):
        logging.info(f"on_ws_client_disconnected client:{websocket.client}")
        if self.task is not None:
            await self.task.queue_frame(EndFrame())

    async def on_rtc_client_connected(
        self,
        transport: SmallWebRTCTransport,
        connection: SmallWebRTCConnection,
    ):
        logging.info(f"on_rtc_client_connected {connection.pc_id=} {connection.connectionState=}")
        self.session.set_client_id(connection.pc_id)
        message = TransportMessageFrame(
            message={"type": "meta", "protocol": "small-webrtc", "version": "0.0.1"},
            urgent=True,
        )
        await transport.output_processor().send_message(message)

    async def on_rtc_client_disconnected(
        self,
        transport: SmallWebRTCTransport,
        connection: SmallWebRTCConnection,
    ):
        logging.info(f"on_client_disconnected callee id:{connection.pc_id}")
        if self.task is not None:
            await self.task.queue_frame(EndFrame())

    async def on_rtc_app_message(
        self,
        transport: SmallWebRTCTransport,
        connection: SmallWebRTCConnection,
        message: Any,
    ):
        logging.info(f"on_app_message received message: {message}")
