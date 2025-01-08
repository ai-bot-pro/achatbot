import logging
import os

from src.common.types import DailyAudioStreamArgs
from src.common import interface
from src.common.factory import EngineClass, EngineFactory
from src.modules.speech.player.stream_player import PlayStreamInit
from src.types.speech.audio_stream import PyAudioStreamArgs

from dotenv import load_dotenv

load_dotenv(override=True)


class AudioStreamEnvInit:
    @staticmethod
    def getEngine(tag, **kwargs) -> interface.IAudioStream | EngineClass:
        if "pyaudio" in tag:
            from . import pyaudio_stream
        if "daily" in tag:
            from . import daily_room_stream

        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        return engine

    @staticmethod
    def initAudioInStreamEngine(
        tag: str | None = None, kwargs: dict | None = None
    ) -> interface.IAudioStream | EngineClass:
        # audio stream
        tag = tag if tag else os.getenv("AUDIO_IN_STREAM_TAG", "pyaudio_in_stream")
        if kwargs is None:
            kwargs = AudioStreamEnvInit.map_config_func[tag]()
        engine = AudioStreamEnvInit.getEngine(tag, **kwargs)
        logging.info(f"initAudioInStreamEngine: {tag}, {engine}")
        return engine

    @staticmethod
    def initAudioOutStreamEngine(
        tag: str | None = None, kwargs: dict | None = None
    ) -> interface.IAudioStream | EngineClass:
        # audio stream
        tag = tag if tag else os.getenv("AUDIO_OUT_STREAM_TAG", "pyaudio_out_stream")
        if kwargs is None:
            kwargs = AudioStreamEnvInit.map_config_func[tag]()
        engine = AudioStreamEnvInit.getEngine(tag, **kwargs)
        logging.info(f"initAudioOutStreamEngine: {tag}, {engine}")
        return engine

    @staticmethod
    def get_daily_room_audio_in_stream_args() -> dict:
        kwargs = DailyAudioStreamArgs(
            bot_name=os.getenv("BOT_NAME", "chat-bot"),
            input=True,
            output=False,
            in_channels=int(os.getenv("IN_CHANNELS", "1")),
            in_sample_rate=int(os.getenv("IN_SAMPLE_RATE", "16000")),
            in_sample_width=int(os.getenv("IN_SAMPLE_WIDTH", "2")),
            meeting_room_token=os.getenv("MEETING_ROOM_TOKEN", ""),
            meeting_room_url=os.getenv("MEETING_ROOM_URL", ""),
        ).__dict__
        return kwargs

    @staticmethod
    def get_daily_room_audio_out_stream_args() -> dict:
        info = PlayStreamInit.get_stream_info()
        kwargs = DailyAudioStreamArgs(
            bot_name=os.getenv("BOT_NAME", "chat-bot"),
            input=False,
            output=True,
            out_channels=info["channels"],
            out_sample_rate=info["rate"],
            out_sample_width=info["sample_width"],
            meeting_room_token=os.getenv("MEETING_ROOM_TOKEN", ""),
            meeting_room_url=os.getenv("MEETING_ROOM_URL", ""),
        ).__dict__
        return kwargs

    @staticmethod
    def get_pyaudio_in_stream_args() -> dict:
        input_device_index = os.getenv("INPUT_DEVICE_INDEX", None)
        if input_device_index is not None:
            input_device_index = int(input_device_index)
        kwargs = PyAudioStreamArgs(
            input=True,
            output=False,
            input_device_index=input_device_index,
            channels=int(os.getenv("IN_CHANNELS", "1")),
            rate=int(os.getenv("IN_SAMPLE_RATE", "16000")),
            sample_width=int(os.getenv("IN_SAMPLE_WIDTH", "2")),
            format=int(os.getenv("pyaudio_format", "8")),
        ).__dict__
        return kwargs

    @staticmethod
    def get_pyaudio_out_stream_args() -> dict:
        info = PlayStreamInit.get_stream_info()
        output_device_index = os.getenv("OUTPUT_DEVICE_INDEX", None)
        if output_device_index is not None:
            output_device_index = int(output_device_index)
        kwargs = PyAudioStreamArgs(
            input=False,
            output=True,
            output_device_index=output_device_index,
            channels=info["channels"],
            rate=info["rate"],
            sample_width=info["sample_width"],
            format=info["format"],
        ).__dict__

        return kwargs

    # TAG : config
    map_config_func = {
        "pyaudio_in_stream": get_pyaudio_in_stream_args,
        "pyaudio_out_stream": get_pyaudio_out_stream_args,
        "daily_room_audio_in_stream": get_daily_room_audio_in_stream_args,
        "daily_room_audio_out_stream": get_daily_room_audio_out_stream_args,
    }
