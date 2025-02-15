import logging
import os

from src.core.llm import LLMEnvInit
from src.common.types import MODELS_DIR
from src.common import interface
from src.common.factory import EngineClass, EngineFactory

from dotenv import load_dotenv

load_dotenv(override=True)


class ASREnvInit:
    @staticmethod
    def getEngine(tag, **kwargs) -> interface.IAsr | EngineClass:
        if "minicpmo_asr" in tag:
            from . import minicpmo_asr
        if "sense_voice" in tag:
            from . import sense_voice_asr
        elif "groq" in tag:
            from . import whisper_groq_asr
        elif "whisper_" in tag:
            from . import whisper_asr

        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        return engine

    @staticmethod
    def initASREngine(tag: str | None = None, **kwargs) -> interface.IAsr | EngineClass:
        def get_args(tag):
            if tag in ASREnvInit.map_config_func:
                return ASREnvInit.map_config_func[tag]()
            return ASREnvInit.get_asr_args()

        # asr
        tag = tag or os.getenv("ASR_TAG", "whisper_timestamped_asr")
        kwargs = kwargs or get_args(tag)
        logging.info(f"initASREngine: {tag}, {kwargs}")
        engine = ASREnvInit.getEngine(tag, **kwargs)
        logging.info(f"initASREngine: {tag}, {engine}")
        return engine

    @staticmethod
    def get_asr_args() -> dict:
        kwargs = {}
        kwargs["model_name_or_path"] = os.getenv("ASR_MODEL_NAME_OR_PATH", "base")
        kwargs["download_path"] = MODELS_DIR
        kwargs["verbose"] = bool(os.getenv("ASR_VERBOSE", "True"))
        kwargs["language"] = os.getenv("ASR_LANG", "zh")
        return kwargs

    @staticmethod
    def get_asr_minicpmo_args() -> dict:
        kwargs = LLMEnvInit.get_llm_transformers_args()
        kwargs["language"] = os.getenv("ASR_LANG", "zh")
        kwargs["use_gptq_ckpt"] = bool(os.getenv("USE_GPTQ_CKPT", ""))
        return kwargs

    map_config_func = {
        "minicpmo_asr": get_asr_minicpmo_args,
    }
