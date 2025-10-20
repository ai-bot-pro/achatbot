import os
import logging

from src.modules.speech.detector import VADEnvInit
from src.common import interface
from src.common.factory import EngineClass, EngineFactory
from src.common.types import VADAnalyzerArgs

from dotenv import load_dotenv

load_dotenv(override=True)


class VADAnalyzerEnvInit:
    @staticmethod
    def get_vad_analyzer_args() -> dict:
        return VADAnalyzerArgs(
            sample_rate=int(os.getenv("SAMPLE_RATE", "16000")),
            num_channels=int(os.getenv("NUM_CHANNELS", "1")),
            confidence=float(os.getenv("CONFIDENCE", "0.7")),
            start_secs=float(os.getenv("START_SECS", "0.032")),
            stop_secs=float(os.getenv("STOP_SECS", "0.32")),
            min_volume=float(os.getenv("MIN_VOLUME", "0.6")),
        ).__dict__

    @staticmethod
    def get_silero_vad_analyzer_args() -> dict:
        return {**VADAnalyzerEnvInit.get_vad_analyzer_args(), **VADEnvInit.get_silero_vad_args()}

    map_config_func = {
        "silero_vad_analyzer": get_silero_vad_analyzer_args,
        "daily_webrtc_vad_analyzer": get_vad_analyzer_args,
        "ten_vad_analyzer": get_vad_analyzer_args,
    }

    @staticmethod
    def getEngine(tag, **kwargs) -> interface.IVADAnalyzer | EngineClass:
        if "daily_webrtc_vad_analyzer" in tag:
            from . import daily_webrtc
        elif "silero_vad_analyzer" in tag:
            from . import silero
        elif "ten_vad_analyzer" in tag:
            from . import ten

        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        return engine

    @staticmethod
    def initVADAnalyzerEngine(
        tag: str | None = None, kwargs: dict | None = None
    ) -> interface.IVADAnalyzer | EngineClass:
        # vad Analyzer
        # daily_webrtc_vad_analyzer for english, chinese vad don't ok~
        tag = tag or os.getenv("VAD_ANALYZER_TAG", "silero_vad_analyzer")
        kwargs = kwargs or VADAnalyzerEnvInit.map_config_func[tag]()
        engine = VADAnalyzerEnvInit.getEngine(tag, **kwargs)
        logging.info(f"initVADEngine: {tag}, {engine}")
        return engine
