import threading
import logging
import queue
import sys

from daily import CallClient, Daily

from src.common.interface import IAudioStream
from src.common.factory import EngineClass
from src.common.types import DailyAudioStreamArgs
from src.types.speech.audio_stream import AudioStreamInfo


class JoinedClients:
    room_joined_clients = {}

    @staticmethod
    def add_room(url):
        JoinedClients.room_joined_clients[url] = {}

    @staticmethod
    def get_room_clients(url):
        return JoinedClients.room_joined_clients.get(url)

    @staticmethod
    def remove_room_clients(url):
        if JoinedClients.room_joined_clients.get(url):
            JoinedClients.room_joined_clients.pop(url)

    @staticmethod
    def add_room_client(url, name, client):
        if JoinedClients.room_joined_clients.get(url) is None:
            JoinedClients.add_room(url)
        JoinedClients.room_joined_clients[url][name] = client

    @staticmethod
    def get_room_client(url, name):
        if JoinedClients.room_joined_clients.get(url):
            return JoinedClients.room_joined_clients.get(url).get(name)
        return None

    @staticmethod
    def remove_room_client(url, name):
        if JoinedClients.room_joined_clients.get(url) and JoinedClients.room_joined_clients.get(
            url
        ).get(name):
            JoinedClients.room_joined_clients[url].pop(name)


class DailyRoomAudioStream(EngineClass, IAudioStream):
    """
    !NOTE: need create daily room
    """

    TAG = [
        "daily_room_audio_stream",
        "daily_room_audio_in_stream",
        "daily_room_audio_out_stream",
    ]
    is_daily_init = False
    is_client_released = False

    def __init__(self, **args):
        self.args = DailyAudioStreamArgs(**args)
        if self.args.input is False and self.args.output is False:
            logging.warning("input and output don't all be False")

        if DailyRoomAudioStream.is_daily_init is False:
            # init once
            Daily.init()
            DailyRoomAudioStream.is_daily_init = True

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
                sample_rate=self.args.out_sample_rate,
                channels=self.args.out_channels,
            )

        self._client: CallClient = None

        self._join_event = threading.Event()
        self._stream_quit = False
        self._app_error = None

        self._in_thread = None
        self._out_thread = None

        self._read_bytes = bytearray()

    def _join(self):
        logging.info(f"{self.args.bot_name} joining")
        self._client.update_subscription_profiles(
            {
                "base": {
                    "screenVideo": "unsubscribed",
                    "camera": "unsubscribed",
                    "microphone": "subscribed",
                }
            }
        )
        self._client.set_user_name(self.args.bot_name)
        self._client.join(
            self.args.meeting_room_url,
            self.args.meeting_room_token,
            client_settings={
                "inputs": {
                    "camera": False,
                    "microphone": {
                        "isEnabled": self.args.output,
                        "settings": {"deviceId": "my-mic"},
                    },
                }
            },
            completion=self.on_joined,
        )
        self._join_event.wait()

    def on_joined(self, data, error):
        if error:
            logging.info(f"Unable to join meeting: {error}")
            self._app_error = error
        else:
            JoinedClients.add_room_client(
                self.args.meeting_room_url, self.args.bot_name, self._client
            )
            logging.info(f"join ok, joined meeting data: {data}")

        self._join_event.set()

    def open_stream(self):
        client = JoinedClients.get_room_client(self.args.meeting_room_url, self.args.bot_name)
        if client is not None:
            logging.info(
                f"{self.args.meeting_room_url} {self.args.bot_name} get joined client {client}"
            )
            self._client = client
            self._join_event.set()
            return

        self._client: CallClient = CallClient()
        DailyRoomAudioStream.is_client_released = False
        if self._join_event.is_set() is False:
            self._join()

    def start_stream(self):
        self._join_event.wait()
        if self._app_error:
            logging.error("Unable to receive audio!")
            return
        self._stream_quit = False
        if self.args.input and self._in_thread is None:
            self._in_queue = queue.Queue()  # old queue no ref count to gc free
            self._in_thread = threading.Thread(target=self._receive_audio)
            self._in_thread.start()

        if self.args.is_async_output and self._out_thread is None:
            self._out_queue = queue.Queue()
            self._out_thread = threading.Thread(target=self._send_raw_audio)
            self._out_thread.start()

    def stop_stream(self):
        self._stream_quit = True
        if self._in_thread and self._in_thread.is_alive():
            self._in_thread.join()
            self._in_thread = None
            logging.info("in thread stoped")
        if self._out_thread and self._out_thread.is_alive():
            self._out_thread.join()
            self._out_thread = None
            logging.info("out thread stoped")

    def close_stream(self):
        client = JoinedClients.get_room_client(self.args.meeting_room_url, self.args.bot_name)
        if client is None:
            logging.info("client leave, stream already closed")
            return
        self.stop_stream()
        self._join_event.clear()
        self._client.leave()
        JoinedClients.remove_room_client(self.args.meeting_room_url, self.args.bot_name)
        logging.info("stream closed")

    def is_stream_active(self) -> bool:
        if self._join_event.is_set() is False:
            return False
        if self._app_error:
            return False
        if self._stream_quit is True:
            return False
        if self.args.input:
            return self._in_thread is not None and self._in_thread.is_alive()
        if self.args.is_async_output:
            return self._out_thread is not None and self._out_thread.is_alive()
        if self.args.output:
            return True

    def close(self) -> None:
        if DailyRoomAudioStream.is_client_released:
            logging.info("client already released")
            return

        self.close_stream()
        self._client.release()
        DailyRoomAudioStream.is_client_released = True
        logging.info("client released")

    def _receive_audio(self):
        self._join_event.wait()

        if self._app_error:
            logging.error("Unable to receive audio!")
            return
        logging.info("start run receive audio!")

        while not self._stream_quit:
            buffer = self._speaker_device.read_frames(self.args.frames_per_buffer)
            if len(buffer) > 0:
                if self.args.stream_callback:
                    self.args.stream_callback(buffer, self.args.frames_per_buffer)
                    continue
                if self.args.input:  # put to queue
                    self._in_queue.put(buffer)
                else:  # write to stdout pipe stream
                    sys.stdout.buffer.write(buffer)

        logging.info("stop run receive audio!")

    def get_stream_info(self) -> AudioStreamInfo:
        return AudioStreamInfo(
            in_channels=self.args.in_channels,
            in_sample_rate=self.args.in_sample_rate,
            in_sample_width=self.args.in_sample_width,
            in_frames_per_buffer=self.args.frames_per_buffer,
            out_channels=self.args.out_channels,
            out_sample_rate=self.args.out_sample_rate,
            out_sample_width=self.args.out_sample_width,
            out_frames_per_buffer=self.args.frames_per_buffer,
            pyaudio_out_format=None,
        )

    def read_stream(self, num_frames) -> bytes:
        if self.args.stream_callback:
            return b""
        if self.args.input and self._in_queue:
            num_read_bytes = num_frames * self.args.in_channels * self.args.in_sample_width
            while len(self._read_bytes) < num_read_bytes:
                read_bytes = self._in_queue.get(block=True)
                self._read_bytes.extend(read_bytes)
            return_bytes = self._read_bytes[:num_read_bytes]
            self._read_bytes = self._read_bytes[num_read_bytes:]
            return bytes(return_bytes)

        if num_frames < 0:
            raise ValueError("num_frames must be non-negative")
        if num_frames == 0:
            num_frames = self.args.frames_per_buffer
        return self._speaker_device.read_frames(num_frames)

    def write_stream(self, data):
        if self.args.is_async_output:  # async write frames
            self._out_queue.put(data)
        else:
            data and self._mic_device.write_frames(data)

    def _send_raw_audio(self):
        self._join_event.wait()

        if self._app_error:
            logging.error("Unable to send audio!")
            return
        logging.info("start run send audio!")

        while not self._stream_quit:
            if self.args.output:  # from queue which write_stream put buffer
                try:
                    buffer = self._out_queue.get(block=True, timeout=self.args.out_queue_timeout_s)
                except queue.Empty:
                    logging.debug(
                        f"tts_synthesize's consumption queue is empty after block {self.args.out_queue_timeout_s}s"
                    )
                    continue
            else:  # from stdin pipe stream
                num_bytes = self.args.frames_per_buffer * self.args.out_channels * 2
                buffer = sys.stdin.buffer.read(num_bytes)
            buffer and self._mic_device.write_frames(buffer)

        logging.info("stop run send audio!")
