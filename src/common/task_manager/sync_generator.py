import asyncio
from typing import Any, AsyncGenerator, Generator

import concurrent


class SynchronizedGenerator(Generator[Any, None, None]):
    def __init__(
        self,
        generator: AsyncGenerator[Any, None],
        loop: asyncio.AbstractEventLoop,
    ):
        self._generator = generator
        self._loop = loop

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return asyncio.run_coroutine_threadsafe(
                self._generator.__anext__(),
                self._loop,
            ).result()
        except StopAsyncIteration as e:
            raise StopIteration from e
        except concurrent.futures._base.CancelledError as e:
            raise StopIteration from e

    def send(self, value):
        return self.__next__()

    def throw(self, type, value=None, traceback=None):
        raise StopIteration
