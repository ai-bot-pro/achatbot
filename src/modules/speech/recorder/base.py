from queue import Queue
from typing import Generator, AsyncGenerator

import pyaudio

from src.common.factory import EngineClass
from src.common.types import AudioRecoderArgs


class AudioRecorder(EngineClass):
    buffer_queue = Queue()

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**AudioRecoderArgs().__dict__, **kwargs}

    @classmethod
    def stream_callback(cls, in_data, frame_count=0, time_info=None, status=None):
        # print(len(in_data), frame_count, time_info, status, cls.buffer_queue and cls.buffer_queue.qsize())
        cls.buffer_queue and cls.buffer_queue.put_nowait(in_data)
        play_data = chr(0) * len(in_data)
        return play_data, pyaudio.paContinue

    def __init__(self, **args) -> None:
        self.args = AudioRecoderArgs(**args)
        if self.args.audio_stream is None:
            raise Exception("audio_stream is None")
        stream_callback = AudioRecorder.stream_callback if self.args.is_stream_callback else None
        self.args.audio_stream.set_args(
            stream_callback=stream_callback,
        )
        self.audio = self.args.audio_stream
        # self.audio = PyAudioStream(PyAudioStreamArgs(
        #    format=self.args.format,
        #    channels=self.args.channels,
        #    rate=self.args.rate,
        #    input_device_index=self.args.input_device_index,
        #    input=True,
        #    frames_per_buffer=self.args.frames_per_buffer,
        #    stream_callback=stream_callback,
        # ))

    def get_record_buf(self) -> bytes:
        if self.args.is_stream_callback is False:
            return self.audio.read_stream(self.args.num_frames)
        return self.buffer_queue.get()

    def frame_genrator(self) -> Generator[bytes, None, None]:
        while True:
            yield self.get_record_buf()

    def close(self):
        self.audio.close()
