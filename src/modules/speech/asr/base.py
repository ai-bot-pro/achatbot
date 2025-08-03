import os
from queue import Queue
from typing import Generator
import threading
from concurrent.futures import ThreadPoolExecutor

from src.common.session import Session
from src.common.factory import EngineClass
from src.types.speech.asr.base import ASRArgs
from src.common.utils import task
from src.common.interface import IAsr
from src.common.utils.audio_utils import bytes2NpArrayWith16, read_wav_to_np, resample_audio
from src.common.types import RATE


class ASRBase(EngineClass, IAsr):
    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**ASRArgs().__dict__, **kwargs}

    def __init__(self, **args) -> None:
        self.args = ASRArgs(**args)
        self.asr_audio = None
        # threading
        self.lock = threading.Lock()

    def set_audio_data(self, audio_data):
        self.lock.acquire()
        if isinstance(audio_data, (bytes, bytearray)):
            self.asr_audio = bytes2NpArrayWith16(audio_data)
        elif (
            isinstance(audio_data, str)
            and audio_data.endswith(".wav")
            and os.path.exists(audio_data)
        ):
            self.asr_audio, sr = read_wav_to_np(audio_data)
            if sr != RATE:
                self.asr_audio = resample_audio(self.asr_audio, sr, RATE)
        elif isinstance(audio_data, str):
            self.asr_audio = audio_data
        self.lock.release()
        return

    def transcribe_stream_sync(self, session: Session) -> Generator[str, None, None]:
        queue: Queue = Queue()
        with ThreadPoolExecutor() as executor:
            executor.submit(task.fetch_async_items, queue, self.transcribe_stream, session)
            while True:
                item = queue.get()
                if item is None:
                    break
                yield item
