from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Generator

from src.common.session import Session
from src.common.factory import EngineClass
from src.common.types import WhisperASRArgs
from src.common.utils import task
from src.common.interface import IAsr


class WhisperASRBase(EngineClass, IAsr):
    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**WhisperASRArgs().__dict__, **kwargs}

    def __init__(self, **args) -> None:
        self.args = WhisperASRArgs(**args)
        self.asr_audio = None

    def transcribe_stream_sync(self, session: Session) -> Generator[str, None, None]:
        queue: Queue = Queue()
        with ThreadPoolExecutor() as executor:
            executor.submit(task.fetch_async_items, queue, self.transcribe_stream, session)
            while True:
                item = queue.get()
                if item is None:
                    break
                yield item
