import logging

from fastapi import WebSocket
from apipeline.frames.control_frames import EndFrame

from src.cmd.bots.base import AIBot
from src.transports.fastapi_websocket_server import FastapiWebsocketTransport


class AIFastapiWebsocketBot(AIBot):
    def __init__(self, websocket: WebSocket | None = None, **args) -> None:
        super().__init__(**args)
        self._websocket = websocket

    def set_fastapi_websocket(self, websocket: WebSocket):
        self._websocket = websocket

    async def on_client_connected(
        self,
        transport: FastapiWebsocketTransport,
        websocket: WebSocket,
    ):
        logging.info(f"on_client_connected client:{websocket.client}")
        self.session.set_client_id(client_id=f"{websocket.client.host}:{websocket.client.port}")

    async def on_client_disconnected(
        self,
        transport: FastapiWebsocketTransport,
        websocket: WebSocket,
    ):
        logging.info(f"on_client_disconnected client:{websocket.client}")
        if self.task is not None:
            await self.task.queue_frame(EndFrame())
