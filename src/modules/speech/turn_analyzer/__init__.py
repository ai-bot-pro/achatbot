import os
import logging


from dotenv import load_dotenv

from src.common import interface
from src.common.factory import EngineClass, EngineFactory
from src.types.speech.turn_analyzer.smart_turn import (
    MODELS_DIR,
    SmartTurnArgs,
)

load_dotenv(override=True)


class TurnAnalyzerEnvInit:
    @staticmethod
    def get_smart_turn_analyzer_args() -> dict:
        args = dict(
            **SmartTurnArgs(
                model_path=os.getenv(
                    "TURN_MODEL_PATH", os.path.join(MODELS_DIR, "pipecat-ai/smart-turn-v2")
                ),
                sample_rate=int(os.getenv("TURN_MODEL_PATH", "16000")),
                stop_secs=int(os.getenv("TURN_MODEL_PATH", "3")),
                pre_speech_ms=int(os.getenv("TURN_MODEL_PATH", "0")),
                max_duration_secs=int(os.getenv("TURN_MODEL_PATH", "8")),
            ).__dict__
        )
        return args

    @staticmethod
    def getEngine(tag, **kwargs) -> interface.ITurnAnalyzer | EngineClass:
        if "v2_smart_turn_analyzer" in tag:
            from . import smart_turn

        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        return engine

    @staticmethod
    def initTurnAnalyzerEngine(
        tag: str | None = None, kwargs: dict | None = None
    ) -> interface.ITurnAnalyzer | EngineClass:
        tag = tag or os.getenv("TURN_ANALYZER_TAG", "v2_smart_turn_analyzer")
        kwargs = kwargs or TurnAnalyzerEnvInit.map_config_func[tag]()
        logging.info(f"initTurnAnalyzerEngine: {tag}, {kwargs}")
        engine = TurnAnalyzerEnvInit.getEngine(tag, **kwargs)
        logging.info(f"initTurnAnalyzerEngine: {tag}, {engine}")
        return engine

    map_config_func = {
        "v2_smart_turn_analyzer": get_smart_turn_analyzer_args,
    }
