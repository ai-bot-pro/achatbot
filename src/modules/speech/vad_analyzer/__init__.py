import os
import logging

from src.modules.speech.detector import VADEnvInit
from src.common import interface
from src.common.factory import EngineClass, EngineFactory
from src.common.types import VADAnalyzerArgs

from dotenv import load_dotenv
load_dotenv(override=True)


class VADAnalyzerEnvInit():

    @staticmethod
    def get_vad_analyzer_args() -> dict:
        return VADAnalyzerArgs(
            sample_rate=int(os.getenv('SAMPLE_RATE', "16000")),
            num_channels=int(os.getenv('NUM_CHANNELS', "1")),
            confidence=float(os.getenv('CONFIDENCE', "0.7")),
            start_secs=float(os.getenv('START_SECS', "0.2")),
            stop_secs=float(os.getenv('STOP_SECS', "0.8")),
            min_volume=float(os.getenv('MIN_VOLUME', "0.6")),
        ).__dict__

    @staticmethod
    def get_silero_vad_analyzer_args() -> dict:
        return {**VADAnalyzerEnvInit.get_vad_analyzer_args(), **VADEnvInit.get_silero_vad_args()}

    @staticmethod
    def get_daily_webrtc_vad_analyzer_args() -> dict:
        return VADAnalyzerEnvInit.get_vad_analyzer_args()

    map_config_func = {
        'silero_vad_analyzer': get_silero_vad_analyzer_args,
        'webrtc_vad_analyzer': get_daily_webrtc_vad_analyzer_args,
    }

    @staticmethod
    def initVADAnalyzerEngine() -> interface.IVADAnalyzer | EngineClass:
        from . import daily_webrtc
        from . import silero
        # vad Analyzer
        tag = os.getenv('VAD_ANALYZER_TAG', "silero_vad_analyzer")
        kwargs = VADAnalyzerEnvInit.map_config_func[tag]()
        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        logging.info(f"initVADEngine: {tag}, {engine}")
        return engine
