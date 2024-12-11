import logging
import os

from src.common.types import MODELS_DIR
from src.common import interface
from src.common.factory import EngineClass, EngineFactory

from dotenv import load_dotenv

load_dotenv(override=True)


class ASREnvInit:
    @staticmethod
    def getEngine(tag, **kwargs) -> interface.IAsr | EngineClass:
        if "sense_voice" in tag:
            from . import sense_voice_asr
        elif "groq" in tag:
            from . import whisper_groq_asr
        elif "whisper_" in tag:
            from . import whisper_asr

        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        return engine

    @staticmethod
    def initASREngine(
        tag: str | None = None,
        # kwargs: dict | None = None
    ) -> interface.IAsr | EngineClass:
        # asr
        tag = tag or os.getenv("ASR_TAG", "whisper_timestamped_asr")
        kwargs = {}
        kwargs["model_name_or_path"] = os.getenv("ASR_MODEL_NAME_OR_PATH", "base")
        kwargs["download_path"] = MODELS_DIR
        kwargs["verbose"] = bool(os.getenv("ASR_VERBOSE", "True"))
        kwargs["language"] = os.getenv("ASR_LANG", "zh")
        engine = ASREnvInit.getEngine(tag, **kwargs)
        logging.info(f"initASREngine: {tag}, {engine}")
        return engine
