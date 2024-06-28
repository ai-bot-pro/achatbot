import threading
import logging
import queue
import time
import io

import pyaudio
from pydub import AudioSegment

from src.common.audio_stream import (
    AudioStream,
    AudioStreamArgs,
    AudioBufferManager,
)
from src.common.factory import EngineClass
from src.common.interface import IPlayer
from src.common.session import Session
from src.common.types import AudioPlayerArgs


class PyAudioPlayer(EngineClass):
    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**AudioPlayerArgs().__dict__, **kwargs}

    def __init__(self, **args) -> None:
        self.args = AudioPlayerArgs(**args)
        self.audio = AudioStream(AudioStreamArgs(
            format_=self.args.format_,
            channels=self.args.channels,
            rate=self.args.rate,
            output_device_index=self.args.output_device_index,
            output=True,
            frames_per_buffer=self.args.sub_chunk_size,
        ))

    def close(self):
        self.audio.close()


class StreamPlayer(PyAudioPlayer, IPlayer):
    TAG = "stream_player"

    def __init__(self, **args) -> None:
        super().__init__(**args)

        self.first_chunk_played = False
        self.playback_active = False
        self.playback_thread: threading.Thread = None
        self.pause_event: threading.Event = threading.Event()
        self.immediate_stop_event: threading.Event = threading.Event()

        if not self.args.audio_buffer:  # new a queue
            self.args.audio_buffer = queue.Queue()
        self.buffer_manager: AudioBufferManager = AudioBufferManager(
            self.args.audio_buffer)

    def start(self, session: Session):
        self.first_chunk_played = False
        self.playback_active = True
        self.audio.open_stream()
        self.audio.start_stream()

        if not self.playback_thread or not self.playback_thread.is_alive():
            self.playback_thread = threading.Thread(
                target=self._process_buffer, args=(session,))
            self.playback_thread.start()

    def _process_buffer(self, session: Session):
        while self.playback_active:
            try:
                chunk = self.buffer_manager.get_from_buffer()
                chunk and self._play_chunk(session, chunk)

                if self.immediate_stop_event.is_set():
                    logging.info(
                        f"Immediate stop requested, aborting {self.TAG}")
                    break
            except queue.Empty:
                continue
            except Exception as ex:
                raise ex

        self.args.on_play_end and self.args.on_play_end(session)
        self.playback_active = False

    def _play_chunk(self, session: Session, chunk):
        # handle mpeg
        if self.args.format_ == pyaudio.paCustomFormat:
            # convert to pcm using pydub
            segment = AudioSegment.from_mp3(io.BytesIO(chunk))
            chunk = segment.raw_data

        for i in range(0, len(chunk), self.args.sub_chunk_size):
            sub_chunk = chunk[i:i + self.args.sub_chunk_size]

            if not self.first_chunk_played and self.args.on_play_start:
                self.on_play_start(session, sub_chunk)
                self.first_chunk_played = True

            if self.args.on_play_chunk:
                self.args.on_play_chunk(session, sub_chunk)

            self.audio.stream.write(sub_chunk)

            while self.pause_event.is_set():
                time.sleep(0.01)

            if self.immediate_stop_event.is_set():
                break

    def play_audio(self, session: Session):
        """
        send playing audio chunk to buffer (queue.Queue)
        """
        if session.ctx.state.get("tts_chunk") is None:
            return
        self.buffer_manager.add_to_buffer(session.ctx.state["tts_chunk"])

    def stop(self, session: Session):
        if not self.playback_thread:
            logging.warn("No playback thread found, cannot stop playback")
            return

        if self.args.is_immediate_stop:
            self.immediate_stop_event.set()
            # wait immediate stop play audio over
            while self.playback_active:
                time.sleep(0.1)
            self.immediate_stop_event.clear()
            return

        self.playback_active = False

        if self.playback_thread and self.playback_thread.is_alive():
            # wait current audio play over
            self.playback_thread.join()

        time.sleep(0.1)

        self.immediate_stop_event.clear()
        self.buffer_manager.clear_buffer()
        self.playback_thread = None
        self.close()

    def pause(self):
        self.pause_event.set()

    def resume(self):
        self.pause_event.clear()
