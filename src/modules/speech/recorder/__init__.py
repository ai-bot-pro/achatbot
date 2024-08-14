
import logging
import os

from src.common.types import AudioRecoderArgs, VADRecoderArgs
from src.common import interface
from src.common.factory import EngineClass, EngineFactory

from dotenv import load_dotenv
load_dotenv(override=True)


class RecorderEnvInit():

    @staticmethod
    def initRecorderEngine() -> interface.IRecorder | EngineClass:
        from . import rms_record
        from . import vad_record
        # recorder
        tag = os.getenv('RECORDER_TAG', "rms_recorder")
        kwargs = RecorderEnvInit.map_config_func[tag]()
        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        logging.info(f"initRecorderEngine: {tag}, {engine}")
        return engine

    @staticmethod
    def get_vad_recorder_args() -> dict:
        kwargs = VADRecoderArgs(
            is_stream_callback=bool(os.getenv('IS_STREAM_CALLBACK', "True")),
        ).__dict__
        return kwargs

    @staticmethod
    def get_rms_recorder_args() -> dict:
        kwargs = AudioRecoderArgs(
            is_stream_callback=bool(os.getenv('IS_STREAM_CALLBACK', "True")),
        ).__dict__
        return kwargs

    # TAG : config
    map_config_func = {
        'rms_recorder': get_rms_recorder_args,
        'wakeword_rms_recorder': get_rms_recorder_args,
        'vad_recorder': get_vad_recorder_args,
        'wakeword_vad_recorder': get_vad_recorder_args,
    }
