import os
import logging

from src.common import interface
from src.common.factory import EngineClass, EngineFactory
from src.common.types import AudioPlayerArgs

from dotenv import load_dotenv

load_dotenv(override=True)


class PlayerEnvInit:
    @staticmethod
    def initPlayerEngine() -> interface.IPlayer | EngineClass:
        from . import stream_player

        # player
        tag = os.getenv("PLAYER_TAG", "stream_player")
        kwargs = AudioPlayerArgs(
            is_immediate_stop=bool(os.getenv("IS_IMMEDIATE_STOP", "")),
        ).__dict__
        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        logging.info(f"initPlayerEngine: {tag},  {engine}")
        return engine
