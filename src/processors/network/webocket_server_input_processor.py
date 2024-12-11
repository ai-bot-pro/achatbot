import logging
import asyncio
import websockets

from apipeline.frames.data_frames import AudioRawFrame
from apipeline.frames.control_frames import StartFrame, EndFrame
from apipeline.frames.sys_frames import CancelFrame

from src.processors.audio_input_processor import AudioVADInputProcessor
from src.types.network.websocket import WebsocketServerCallbacks, WebsocketServerParams


class WebsocketServerInputProcessor(AudioVADInputProcessor):
    def __init__(
        self,
        host: str,
        port: int,
        params: WebsocketServerParams,
        callbacks: WebsocketServerCallbacks,
        **kwargs,
    ):
        super().__init__(params, **kwargs)

        self._host = host
        self._port = port
        self._params = params
        self._callbacks = callbacks

        self._websocket: websockets.WebSocketServerProtocol | None = None

        self._stop_server_event = asyncio.Event()

    async def start(self, frame: StartFrame):
        self._server_task = self.get_event_loop().create_task(self._server_task_handler())
        await super().start(frame)

    async def stop(self):
        await super().stop()
        self._stop_server_event.set()
        await self._server_task
        print("-----websocket server input stop------")

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        self._stop_server_event.set()
        await self._server_task

    async def _server_task_handler(self):
        logging.info(f"Starting websocket server on {self._host}:{self._port}")
        async with websockets.serve(self._client_handler, self._host, self._port):
            await self._stop_server_event.wait()

    async def _client_handler(self, websocket: websockets.WebSocketServerProtocol, path):
        logging.info(f"New client connection from {websocket.remote_address}, path:{path}")
        if self._websocket:
            await self._websocket.close()
            logging.warning("Only one client connected, using new connection")

        self._websocket = websocket

        # Notify
        await self._callbacks.on_client_connected(websocket)

        # Handle incoming messages
        async for message in websocket:
            frame = self._params.serializer.deserialize(message)

            if not frame:
                continue

            if isinstance(frame, AudioRawFrame):
                await self.push_audio_frame(frame)
            else:
                await self.push_frame(frame)

        # Notify disconnection
        await self._callbacks.on_client_disconnected(websocket)

        await self._websocket.close()
        self._websocket = None

        logging.info(f"Client {websocket.remote_address} disconnected")
