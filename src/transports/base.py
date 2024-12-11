from abc import ABC, abstractmethod
import asyncio

from apipeline.processors.frame_processor import FrameProcessor

from src.common.event import EventHandlerManager


class BaseTransport(EventHandlerManager, ABC):
    def __init__(
        self,
        input_name: str | None = None,
        output_name: str | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        super().__init__()
        self._input_name = input_name
        self._output_name = output_name
        self._loop = loop or asyncio.get_running_loop()

    @abstractmethod
    def input_processor(self) -> FrameProcessor:
        raise NotImplementedError

    @abstractmethod
    def output_processor(self) -> FrameProcessor:
        raise NotImplementedError
