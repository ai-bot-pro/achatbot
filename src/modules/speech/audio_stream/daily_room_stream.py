import threading
import logging
import queue
import sys

from daily import (
    CallClient,
    Daily
)

from src.common.interface import IAudioStream
from src.common.factory import EngineClass
from src.common.types import DailyAudioStreamArgs


class JoinedClients():
    room_joined_clients = {}

    @staticmethod
    def add_room(url):
        JoinedClients.room_joined_clients[url] = {}

    @staticmethod
    def add_room_client(url, name, client):
        if JoinedClients.room_joined_clients.get(url) is None:
            JoinedClients.regiter_room(url)
        JoinedClients.room_joined_clients[url][name] = client

    @staticmethod
    def remove_room_client(url, name):
        if JoinedClients.room_joined_clients.get(url) and \
                JoinedClients.room_joined_clients.get(url).get(name):
            JoinedClients.room_joined_clients[url].pop(name)


class DailyRoomAudioStream(EngineClass, IAudioStream):
    TAG = [
        "daily_room_audio_stream",
        "daily_room_audio_in_stream",
        "daily_room_audio_out_stream",
    ]

    def __init__(self, **args):
        self.args = DailyAudioStreamArgs(**args)
        if self.args.input is False and self.args.input is False:
            logging.warning("input and output don't all be False")

        if self.args.input:
            self._speaker_device = Daily.create_speaker_device(
                "my-speaker",
                sample_rate=self.args.in_sample_rate,
                channels=self.args.in_channels,
            )
            Daily.select_speaker_device("my-speaker")

        if self.args.output:
            self._mic_device = Daily.create_microphone_device(
                "my-mic",
                sample_rate=self.args.in_sample_rate,
                channels=self.args.in_channels,
            )

        if self.args.input or self.args.output:
            self._client = CallClient()
            self._client.update_subscription_profiles({
                "base": {
                    "screenVideo": "unsubscribed",
                    "camera": "unsubscribed",
                    "microphone": "subscribed"
                }
            })

        self._join_event = threading.Event()
        self._app_quit = False
        self._app_error = None

        self._in_thread = None
        self._out_thread = None

    def _join(self):
        logging.info(f"{self._bot_name} joining")
        self._client.set_user_name(self._bot_name)
        self._client.join(
            self.args.meeting_room_url,
            self.args.meeting_room_token,
            client_settings={
                "inputs": {
                    "camera": False,
                    "microphone": {
                        "isEnabled": self.args.output,
                        "settings": {
                            "deviceId": "my-mic"
                        }
                    }
                }
            },
            completion=self.on_joined,
        )

    def on_joined(self, data, error):
        if error:
            logging.info(f"Unable to join meeting: {error}")
            self._app_error = error
        else:
            logging.info(f"join ok, joined meeting data: {data}")

        self._join_event.set()

    def open_stream(self):
        if self._join_event.is_set() is False:
            self._join()
            if self._app_error:
                return False
            JoinedClients.add_room_client(
                self.args.meeting_room_url,
                self.args.bot_name,
                self._client
            )

    def start_stream(self):
        if self._input and self._in_thread is None:
            self._in_queue = queue.Queue()  # old queue no ref count to gc free
            self._in_thread = threading.Thread(target=self._receive_audio)
            self._in_thread.start()

        if self.args.output and self._out_thread is None:
            self._out_queue = queue.Queue()
            self._out_thread = threading.Thread(target=self._send_raw_audio)
            self._out_thread.start()

    def stop_stream(self):
        self._app_quit = True
        if self._in_thread and self._in_thread.is_alive():
            self._in_thread.join()
            self._in_thread = None
        if self._out_thread and self._out_thread.is_alive():
            self._out_thread.join()
            self._out_thread = None

    def close_stream(self):
        self.stop_stream()
        self._join_event.clear()
        self._client.leave()
        JoinedClients.remove_room_client(
            self.args.meeting_room_url,
            self.args.bot_name
        )

    def is_stream_active(self) -> bool:
        if self.args.input:
            return self._in_thread and self._in_thread.is_alive()
        if self.args.output:
            return self._out_thread and self._out_thread.is_alive()

    def close(self) -> None:
        if self._app_quit is False:
            self.close_stream()
        self._client.release()

    def _receive_audio(self):
        self._join_event.wait()

        if self._app_error:
            logging.error(f"Unable to receive audio!")
            return

        while not self._app_quit:
            # Read 100ms worth of audio frames.
            buffer = self._speaker_device.read_frames(
                int(self.args.in_sample_rate / 10))
            if len(buffer) > 0:
                if self.args.stream_callback:
                    self.args.stream_callback(buffer)
                    continue
                if self._input:  # put to queue
                    self._in_queue.put(buffer)
                else:  # write to stdout pipe stream
                    sys.stdout.buffer.write(buffer)

    def read_stream(self, num_frames) -> bytes:
        if self.args.stream_callback:
            return b""
        if self.args.input and self._in_queue:
            return self._in_queue.get(block=True, timeout=1)

        if num_frames < 0:
            raise ValueError("num_frames must be non-negative")
        if num_frames == 0:
            num_frames = int(self.args.in_sample_rate / 10)
        return self._speaker_device.read_frames(num_frames)

    def write_stream(self, data):
        if self._out_queue:
            self._out_queue.put(data)

    def _send_raw_audio(self):
        self._join_event.wait()

        if self._app_error:
            logging.error(f"Unable to send audio!")
            return

        while not self._app_quit:
            if self._output:  # from queue which write_stream put buffer
                buffer = self._out_queue.get(block=True, timeout=1)
            else:  # from stdin pipe stream
                num_bytes = int(self.args.sample_rate / 10) * \
                    self.__num_channels * 2
                buffer = sys.stdin.buffer.read(num_bytes)
            buffer and self._mic_device.write_frames(buffer)
