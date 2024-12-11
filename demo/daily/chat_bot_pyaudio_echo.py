import argparse
import threading
import asyncio
import aiohttp
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


class ReceiveDailyAudio:
    def __init__(self, sample_rate, num_channels, is_echo=False, bot_name="chat_bot_pyaudio_echo"):
        self.__sample_rate = sample_rate
        self._bot_name = bot_name

        self.__speaker_device = Daily.create_speaker_device(
            "my-speaker", sample_rate=sample_rate, channels=num_channels
        )
        Daily.select_speaker_device("my-speaker")

        self.__client = CallClient()
        self.__client.update_subscription_profiles(
            {
                "base": {
                    "screenVideo": "unsubscribed",
                    "camera": "unsubscribed",
                    # "microphone": "subscribed"
                }
            }
        )

        self.__app_quit = False
        self.__app_error = None

        self.__start_event = threading.Event()
        self.__thread = threading.Thread(target=self.receive_audio)
        self.__thread.start()

        self._is_echo = is_echo
        if is_echo:
            import uuid
            from src.cmd.init import Env
            from src.common.types import SessionCtx
            from src.common.session import Session

            os.environ["TTS_TAG"] = "tts_daily_speaker"
            self.player = Env.initPlayerEngine()
            client_id = str(uuid.uuid4())
            self.session = Session(**SessionCtx(client_id).__dict__)
            self.player.start(self.session)

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
        self.__client.join(meeting_url, meeting_token, completion=self.on_joined)
        self.__thread.join()

    def leave(self):
        self.__app_quit = True
        self.__thread.join()
        self.__client.leave()
        self.__client.release()
        if self._is_echo:
            self.player.stop(self.session)
            self.player.close()

    def receive_audio(self):
        self.__start_event.wait()

        if self.__app_error:
            print("Unable to receive audio!")
            return

        while not self.__app_quit:
            # Read 100ms worth of audio frames.
            buffer = self.__speaker_device.read_frames(int(self.__sample_rate / 10))
            # print(len(buffer), self._is_echo)
            if len(buffer) > 0:
                if self._is_echo:
                    self.session.ctx.state["tts_chunk"] = buffer
                    self.player.play_audio(self.session)
                else:
                    sys.stdout.buffer.write(buffer)


def run_app(args):
    Daily.init()
    app = ReceiveDailyAudio(args.rate, args.channels, args.echo != 0, args.name)
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
    parser.add_argument("-e", "--echo", type=int, default=0, help="is echo audio")
    parser.add_argument(
        "--name", type=str, default="chat_bot_pyaudio_echo", help="enable send audio"
    )

    args = parser.parse_args()
    run_app(args)
