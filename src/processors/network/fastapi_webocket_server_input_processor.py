import logging

from fastapi import WebSocket
from fastapi.websockets import WebSocketState
from apipeline.frames.data_frames import AudioRawFrame
from apipeline.frames.control_frames import StartFrame, EndFrame
from apipeline.frames.sys_frames import CancelFrame

from src.processors.audio_input_processor import AudioVADInputProcessor
from src.types.network.fastapi_websocket import (
    FastapiWebsocketServerCallbacks,
    FastapiWebsocketServerParams,
)


class FastapiWebsocketServerInputProcessor(AudioVADInputProcessor):
    def __init__(
        self,
        websocket: WebSocket,
        params: FastapiWebsocketServerParams,
        callbacks: FastapiWebsocketServerCallbacks,
        **kwargs,
    ):
        super().__init__(params, **kwargs)

        self._websocket = websocket
        self._params = params
        self._callbacks = callbacks

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._callbacks.on_client_connected(self._websocket)
        self._receive_task = self.get_event_loop().create_task(self._receive_messages())

    async def stop(self):
        await super().stop()
        if self._websocket.client_state != WebSocketState.DISCONNECTED:
            await self._websocket.close()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        if self._websocket.client_state != WebSocketState.DISCONNECTED:
            await self._websocket.close()

    async def _receive_messages(self):
        try:
            # async for message in self._websocket.iter_text():
            # async for message in self._websocket.iter_json():
            async for message in self._websocket.iter_bytes():
                frame = self._params.serializer.deserialize(message)

                if not frame:
                    continue

                if isinstance(frame, AudioRawFrame):
                    await self.push_audio_frame(frame)

        except Exception as e:
            logging.error(f"receive_messages error: {e}")
            return

        logging.info(f"{self._websocket=} disconnected")
        await self._callbacks.on_client_disconnected(self._websocket)
