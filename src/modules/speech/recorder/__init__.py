import logging
import os

from src.common.types import AudioRecoderArgs, VADRecoderArgs
from src.common import interface
from src.common.factory import EngineClass, EngineFactory

from dotenv import load_dotenv

load_dotenv(override=True)


class RecorderEnvInit:
    @staticmethod
    def getEngine(tag, **kwargs) -> interface.IRecorder | EngineClass:
        if "rms_recorder" in tag:
            from . import rms_record
        elif "vad_recorder" in tag:
            from . import vad_record
        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        return engine

    @staticmethod
    def initRecorderEngine() -> interface.IRecorder | EngineClass:
        # recorder
        tag = os.getenv("RECORDER_TAG", "vad_recorder")
        kwargs = RecorderEnvInit.map_config_func[tag]()
        engine = RecorderEnvInit.getEngine(tag, **kwargs)
        logging.info(f"initRecorderEngine: {tag}, {engine}")
        return engine

    @staticmethod
    def get_vad_recorder_args() -> dict:
        kwargs = VADRecoderArgs(
            is_stream_callback=bool(os.getenv("IS_STREAM_CALLBACK", "True")),
        ).__dict__
        return kwargs

    @staticmethod
    def get_rms_recorder_args() -> dict:
        kwargs = AudioRecoderArgs(
            is_stream_callback=bool(os.getenv("IS_STREAM_CALLBACK", "True")),
        ).__dict__
        return kwargs

    # TAG : config
    map_config_func = {
        "rms_recorder": get_rms_recorder_args,
        "wakeword_rms_recorder": get_rms_recorder_args,
        "vad_recorder": get_vad_recorder_args,
        "wakeword_vad_recorder": get_vad_recorder_args,
    }
