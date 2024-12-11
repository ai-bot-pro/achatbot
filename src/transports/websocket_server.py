import logging
import asyncio


try:
    import websockets
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error("In order to use websockets, you need to `pip install achatbot[websocket]`.")
    raise Exception(f"Missing module: {e}")

from src.processors.network.webocket_server_input_processor import WebsocketServerInputProcessor
from src.processors.network.webocket_server_output_processor import WebsocketServerOutputProcessor
from src.transports.base import BaseTransport
from src.types.network.websocket import WebsocketServerCallbacks, WebsocketServerParams


class WebsocketServerTransport(BaseTransport):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        params: WebsocketServerParams = WebsocketServerParams(),
        input_name: str | None = None,
        output_name: str | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        super().__init__(input_name=input_name, output_name=output_name, loop=loop)
        self._host = host
        self._port = port
        self._params = params

        self._callbacks = WebsocketServerCallbacks(
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
        )
        self._input_processor: WebsocketServerInputProcessor | None = None
        self._output_processor: WebsocketServerOutputProcessor | None = None
        self._websocket: websockets.WebSocketServerProtocol | None = None

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")

    def input_processor(self) -> WebsocketServerInputProcessor:
        if not self._input_processor:
            self._input_processor = WebsocketServerInputProcessor(
                self._host, self._port, self._params, self._callbacks, name=self._input_name
            )
        return self._input_processor

    def output_processor(self) -> WebsocketServerOutputProcessor:
        if not self._output_processor:
            self._output_processor = WebsocketServerOutputProcessor(
                self._params, name=self._output_name
            )
        return self._output_processor

    async def _on_client_connected(self, websocket):
        if self._output_processor:
            await self._output_processor.set_client_connection(websocket)
            await self._call_event_handler("on_client_connected", websocket)
        else:
            logging.error("A WebsocketServerTransport output is missing in the pipeline")

    async def _on_client_disconnected(self, websocket):
        if self._output_processor:
            await self._output_processor.set_client_connection(None)
            await self._call_event_handler("on_client_disconnected", websocket)
        else:
            logging.error("A WebsocketServerTransport output is missing in the pipeline")
