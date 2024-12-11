import argparse
import threading
import asyncio
import aiohttp
import queue
import os
import sys


from daily import (
    CallClient,
    Daily,
    EventHandler,
    VirtualCameraDevice,
    VirtualMicrophoneDevice,
    VirtualSpeakerDevice,
)


SAMPLE_RATE = 16000
NUM_CHANNELS = 1
BYTES_PER_SAMPLE = 2


class ReceiveSendDailyAudio:
    def __init__(
        self,
        sample_rate,
        num_channels,
        enable_receive=True,
        enable_send=True,
        bot_name="chat_bot_daily_echo",
    ):
        if enable_receive is False and enable_receive is False:
            raise ValueError("enable_receive and enable_send don't all be False")

        self._bot_name = bot_name
        self.__sample_rate = sample_rate
        self.__num_channels = num_channels

        self.__speaker_device = Daily.create_speaker_device(
            "my-speaker", sample_rate=sample_rate, channels=num_channels
        )
        Daily.select_speaker_device("my-speaker")

        self.__mic_device = Daily.create_microphone_device(
            "my-mic", sample_rate=sample_rate, channels=num_channels
        )

        self.__client = CallClient()
        self.__client.update_subscription_profiles(
            {
                "base": {
                    "screenVideo": "unsubscribed",
                    "camera": "unsubscribed",
                    "microphone": "subscribed",
                }
            }
        )

        self.__app_quit = False
        self.__app_error = None
        self.__start_event = threading.Event()
        self._enable_receive = enable_receive
        self._enable_send = enable_send

        if enable_receive:
            self._receive_thread = threading.Thread(target=self.receive_audio)
            self._receive_thread.start()

        if enable_send:
            self._send_thread = threading.Thread(target=self.send_raw_audio)
            self._send_thread.start()

        if enable_receive and enable_receive:
            self._queue = queue.Queue()

    def on_joined(self, data, error):
        if error:
            print(f"Unable to join meeting: {error}")
            self.__app_error = error
        else:
            print(f"join ok, joined meeting data: {data}")
        self.__start_event.set()

    def run(self, meeting_url, meeting_token):
        print(f"run {self._bot_name}")
        self.__client.set_user_name(self._bot_name)
        self.__client.join(
            meeting_url,
            meeting_token,
            client_settings={
                "inputs": {
                    "camera": False,
                    "microphone": {
                        "isEnabled": self._enable_send,
                        "settings": {"deviceId": "my-mic"},
                    },
                }
            },
            completion=self.on_joined,
        )
        self._enable_receive and self._receive_thread.join()
        self._enable_send and self._send_thread.join()

    def leave(self):
        self.__app_quit = True
        self._enable_receive and self._receive_thread.join()
        self._enable_send and self._send_thread.join()
        self.__client.leave()
        self.__client.release()

    def receive_audio(self):
        self.__start_event.wait()

        if self.__app_error:
            print("Unable to receive audio!")
            return

        while not self.__app_quit:
            # Read 100ms worth of audio frames.
            num_frames = int(self.__sample_rate / 10)
            # num_frames = 512
            buffer = self.__speaker_device.read_frames(num_frames)
            if len(buffer) > 0:
                if self._enable_send:  # put to queue
                    self._queue.put(buffer)
                else:  # write to stdout pipe stream
                    sys.stdout.buffer.write(buffer)

    def send_raw_audio(self):
        self.__start_event.wait()

        if self.__app_error:
            print("Unable to send audio!")
            return

        while not self.__app_quit:
            if self._enable_receive:  # from queue which receive put buffer
                buffer = self._queue.get(block=True, timeout=1)
            else:  # from stdin pipe stream
                num_bytes = int(self.__sample_rate / 10) * self.__num_channels * BYTES_PER_SAMPLE
                buffer = sys.stdin.buffer.read(num_bytes)
            buffer and self.__mic_device.write_frames(buffer)


def run_app(args):
    Daily.init()
    app = ReceiveSendDailyAudio(
        args.rate,
        args.channels,
        args.receive != 0,
        args.send != 0,
        bot_name=args.name,
    )
    try:
        app.run(args.u, args.t)
    except KeyboardInterrupt:
        print("Ctrl-C detected. Exiting!", file=sys.stderr)
    finally:
        app.leave()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="chat-bot")
    parser.add_argument("-u", type=str, help="Room URL")
    parser.add_argument("-t", type=str, help="Token")
    parser.add_argument(
        "-c", "--channels", type=int, default=NUM_CHANNELS, help="Number of channels"
    )
    parser.add_argument("-r", "--rate", type=int, default=SAMPLE_RATE, help="Sample rate")
    parser.add_argument("--receive", type=int, default=1, help="enable receive audio")
    parser.add_argument("--send", type=int, default=1, help="enable send audio")
    parser.add_argument("--name", type=str, default="chat_bot_daily_echo", help="enable send audio")

    args = parser.parse_args()
    run_app(args)
