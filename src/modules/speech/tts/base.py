import logging
import re
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import AsyncGenerator, Generator

import pyaudio

from src.common.factory import EngineClass
from src.common.session import Session
from src.common.utils import task


class TTSVoice:
    def __init__(self, name, id):
        self.name = name
        self.id = id

    def __repr__(self):
        return f"<TTSVoice(name={self.name} id={self.id})>"


class BaseTTS(EngineClass):
    def synthesize_sync(self, session: Session) -> Generator[bytes, None, None]:
        is_stream = self.args.tts_stream if hasattr(self.args, "tts_stream") else False
        logging.debug(f"is_stream:{is_stream}")
        if is_stream is False:
            buff = bytearray()
            scratch_buff = bytearray()
            stream_info = self.get_stream_info()
            chunk_length_seconds = 3
            if hasattr(self.args, "chunk_length_seconds"):
                chunk_length_seconds = self.args.chunk_length_seconds
            chunk_length_in_bytes = chunk_length_seconds * \
                stream_info["rate"] * stream_info["sample_width"]

        queue: Queue = Queue()
        with ThreadPoolExecutor() as executor:
            executor.submit(task.fetch_async_items, queue, self.synthesize, session)

            while True:
                item = queue.get()
                if is_stream is True:
                    if item is None:
                        break
                    yield item
                else:
                    if item is None:
                        yield bytes(buff)
                        buff.clear()
                        scratch_buff.clear()
                        break
                    if len(buff) > chunk_length_in_bytes:
                        scratch_buff = buff
                        yield bytes(scratch_buff)
                        buff.clear()
                    else:
                        buff.extend(item)

    async def synthesize(self, session: Session) -> AsyncGenerator[bytes, None]:
        if "tts_text_iter" in session.ctx.state:
            for text in session.ctx.state["tts_text_iter"]:
                text: str = self.filter_special_chars(text)
                if len(text.strip()) == 0:
                    continue
                async for chunk in self._inference(session, text):
                    yield chunk
                silence_chunk = self._get_end_silence_chunk(session, text)
                if silence_chunk:
                    yield silence_chunk
        elif "tts_text" in session.ctx.state:
            text = session.ctx.state["tts_text"]
            text = self.filter_special_chars(text)
            if len(text.strip()) > 0:
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

    def get_stream_info(self) -> dict:
        return {
            "format": pyaudio.paInt16,
            "channels": 1,
            "rate": 22050,
            "sample_width": 2,
        }

    def filter_special_chars(self, text: str) -> str:
        # @TODO: use nlp stream process sentence
        special_chars = ".。,，;；!！?？」>》}\\]】\"”'‘)~"
        return self._filter_special_chars(special_chars, text)

    def remove_trailing_special_chars(self, special_chars: str, s: str) -> str:
        pattern = rf'(.*?)([{special_chars}])[{special_chars}]*$'
        return re.sub(pattern, r'\1\2', s)

    def _filter_special_chars(self, special_chars: str, text: str) -> str:
        pattern = rf'[{special_chars}]'
        match = re.search(pattern, text)
        res = text
        if match:
            first_special_index = match.start()
            res = text[:first_special_index + 1] + \
                re.sub(rf'[{special_chars}]+$', '', text[first_special_index + 1:])
            if len(res) == 1:  # have a special char, return empty
                res = ""
        res = res.strip('\n')
        logging.debug(f"text:{text} --filter--> res:{res} with match:{match}")
        return res
