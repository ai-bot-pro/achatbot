import asyncio
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import AsyncGenerator, Iterator, Generator


from src.common.factory import EngineClass
from src.common.session import Session


class TTSVoice:
    def __init__(self, name, id):
        self.name = name
        self.id = id

    def __repr__(self):
        return f"<TTSVoice(name={self.name} id={self.id})>"


class BaseTTS(EngineClass):
    def synthesize_sync(self, session: Session) -> Generator[bytes, None, None]:
        def fetch_async_items(queue: Queue, ss: Session) -> None:  # type: ignore
            async def get_items() -> None:
                async for item in self.synthesize(ss):
                    queue.put(item)
                queue.put(None)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(get_items())
            loop.close()

        queue: Queue = Queue()  # type: ignore

        with ThreadPoolExecutor() as executor:
            executor.submit(fetch_async_items, queue, session)
            while True:
                item = queue.get()
                if item is None:
                    break
                yield item

    async def synthesize(self, session: Session) -> AsyncGenerator[bytes, None]:
        if "tts_text_iter" in session.ctx.state:
            for text in session.ctx.state["tts_text_iter"]:
                async for chunk in self._inference(session, text):
                    yield chunk
                silence_chunk = self._get_end_silence_chunk(session, text)
                if silence_chunk:
                    yield silence_chunk
        elif "tts_text" in session.ctx.state:
            text = session.ctx.state["tts_text"]
            async for chunk in self._inference(session, text):
                yield chunk
            silence_chunk = self._get_end_silence_chunk(session, text)
            if silence_chunk:
                yield silence_chunk

    async def _inference(self, session: Session, text: str) -> AsyncGenerator[bytes, None]:
        raise NotImplementedError(
            "The _inference method must be implemented by the derived subclass.")

    def _get_end_silence_chunk(self, session: Session, text: str) -> bytes:
        b''
