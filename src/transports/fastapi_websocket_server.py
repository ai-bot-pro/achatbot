import logging
import asyncio

try:
    from fastapi import WebSocket
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error("In order to use fastapi websocket, you need to `pip install achatbot[fastapi]`.")
    raise Exception(f"Missing module: {e}")

from src.processors.network.fastapi_webocket_server_input_processor import (
    FastapiWebsocketServerInputProcessor,
)
from src.processors.network.fastapi_webocket_server_output_processor import (
    FastapiWebsocketServerOutputProcessor,
)
from src.transports.base import BaseTransport
from src.types.network.fastapi_websocket import (
    FastapiWebsocketServerCallbacks,
    FastapiWebsocketServerParams,
)


class FastapiWebsocketTransport(BaseTransport):
    def __init__(
        self,
        websocket: WebSocket,
        params: FastapiWebsocketServerParams,
        input_name: str | None = None,
        output_name: str | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        super().__init__(input_name=input_name, output_name=output_name, loop=loop)
        self._params = params

        self._callbacks = FastapiWebsocketServerCallbacks(
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
        )

        self._input_processor = FastapiWebsocketServerInputProcessor(
            websocket, self._params, self._callbacks, name=self._input_name
        )
        self._output_processor = FastapiWebsocketServerOutputProcessor(
            websocket, self._params, name=self._output_name
        )

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")

    def input_processor(self) -> FastapiWebsocketServerInputProcessor:
        return self._input_processor

    def output_processor(self) -> FastapiWebsocketServerOutputProcessor:
        return self._output_processor

    async def _on_client_connected(self, websocket):
        await self._call_event_handler("on_client_connected", websocket)

    async def _on_client_disconnected(self, websocket):
        await self._call_event_handler("on_client_disconnected", websocket)
