import logging
from queue import Queue, Empty
from typing import Generator, AsyncGenerator

from src.common.factory import EngineClass
from src.common.types import AudioRecoderArgs, PYAUDIO_PACONTINUE
from src.common.interface import IAudioStream
from src.types.speech.audio_stream import AudioStreamInfo


class AudioRecorder(EngineClass):
    buffer_queue = Queue()

    @classmethod
    def stream_callback(cls, in_data, frame_count=0, time_info=None, status=None):
        # print(len(in_data), frame_count, time_info, status, cls.buffer_queue and cls.buffer_queue.qsize())
        cls.buffer_queue and cls.buffer_queue.put_nowait(in_data)
        play_data = chr(0) * len(in_data)
        return play_data, PYAUDIO_PACONTINUE

    def __init__(self, **args) -> None:
        self.args = AudioRecoderArgs(**args)

    def get_record_buf(self) -> bytes:
        if self.args.is_stream_callback is False:
            return self.audio.read_stream(self.args.num_frames)
        # !NOTE: no block for audio check is_stream_active
        return self.buffer_queue.get(timeout=0.1)

    def frame_genrator(self) -> Generator[bytes, None, None]:
        while self.audio.is_stream_active():
            try:
                frames = self.get_record_buf()
                yield frames
            except Empty:
                continue
        logging.info("frame_genrator end")
        # yield None

    def open(self):
        self.audio.open_stream()

    def close(self):
        self.audio.close()

    def set_in_stream(self, audio_stream: EngineClass | IAudioStream):
        stream_callback = AudioRecorder.stream_callback if self.args.is_stream_callback else None
        audio_stream.set_args(
            stream_callback=stream_callback,
        )
        self.audio = audio_stream
        self.stream_info: AudioStreamInfo = self.audio.get_stream_info()
