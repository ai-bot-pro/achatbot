from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Generator, AsyncGenerator

import pyaudio

from src.common.audio_stream import AudioStream, RingBuffer
from src.common.factory import EngineClass
from src.common.session import Session
from src.common.interface import IRecorder
from src.common.utils import task
from src.common.types import AudioRecoderArgs, AudioStreamArgs


class PyAudioRecorder(EngineClass, IRecorder):
    buffer_queue = Queue()

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**AudioRecoderArgs().__dict__, **kwargs}

    @classmethod
    def stream_callback(cls, in_data, frame_count, time_info, status):
        # print(len(in_data), frame_count, time_info, status, cls.buffer_queue and cls.buffer_queue.qsize())
        cls.buffer_queue and cls.buffer_queue.put_nowait(in_data)
        play_data = chr(0) * len(in_data)
        return play_data, pyaudio.paContinue

    def __init__(self, **args) -> None:
        self.args = AudioRecoderArgs(**args)
        stream_callback = PyAudioRecorder.stream_callback if self.args.is_stream_callback else None
        self.audio = AudioStream(AudioStreamArgs(
            format=self.args.format,
            channels=self.args.channels,
            rate=self.args.rate,
            input_device_index=self.args.input_device_index,
            input=True,
            frames_per_buffer=self.args.frames_per_buffer,
            stream_callback=stream_callback,
        ))

    def get_record_buf(self) -> bytes:
        if self.args.is_stream_callback is False:
            return self.audio.stream.read(self.args.num_frames)
        return self.buffer_queue.get()

    def frame_genrator(self) -> Generator[bytes, None, None]:
        while True:
            yield self.get_record_buf()

    def close(self):
        self.audio.close()
