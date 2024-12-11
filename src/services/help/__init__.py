"""
room manager api for admin dashboard or init to run with http api service
"""

import logging
import os

from src.common import interface
from src.common.types import AgoraChannelArgs, DailyRoomArgs, LivekitRoomArgs
from src.common.factory import EngineClass, EngineFactory

from dotenv import load_dotenv

load_dotenv(override=True)


class RoomManagerEnvInit:
    @staticmethod
    def getEngine(tag, **kwargs) -> interface.IRoomManager | EngineClass:
        if "livekit_room" in tag:
            from . import livekit_room
        elif "daily_room" in tag:
            from . import daily_room
        elif "agora_rtc_channel" in tag:
            from .agora import channel

        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        return engine

    @staticmethod
    def initEngine(
        tag: str | None = None, kwargs: dict | None = None
    ) -> interface.IRoomManager | EngineClass:
        tag = tag or os.getenv("ROOM_TAG", "daily_room")
        kwargs = kwargs or RoomManagerEnvInit.map_config_func[tag]()
        engine = RoomManagerEnvInit.getEngine(tag, **kwargs)
        logging.info(f"initEngine: {tag}, {engine}")
        return engine

    @staticmethod
    def get_daily_room_args() -> dict:
        kwargs = DailyRoomArgs(
            privacy=os.getenv("DAILY_ROOM_PRIVACY", "public"),
        ).model_dump()
        return kwargs

    @staticmethod
    def get_livekit_room_args() -> dict:
        kwargs = LivekitRoomArgs(
            bot_name=os.getenv("LIVEKIT_BOT_NAME", "chat-bot"),
            is_common_session=bool(os.getenv("LIVEKIT_IS_COMMON_SESSION", "")),
        ).model_dump()
        return kwargs

    @staticmethod
    def get_agora_channel_args() -> dict:
        kwargs = AgoraChannelArgs().model_dump()
        return kwargs

    # TAG : config
    map_config_func = {
        "daily_room": get_daily_room_args,
        "livekit_room": get_livekit_room_args,
        "agora_rtc_channel": get_agora_channel_args,
    }
