import logging
import os

from src.common import interface
from src.common.factory import EngineClass, EngineFactory


from dotenv import load_dotenv

load_dotenv(override=True)


class SpeechEnhancerEnvInit:
    @staticmethod
    def getEngine(tag, **kwargs) -> interface.ISpeechEnhancer | EngineClass:
        if "enhancer_ans_dfsmn" == tag:
            from . import ans_dfsmn
        elif "enhancer_ans_rnnoise" == tag:
            from . import ans_rnnoise
        elif "enhancer_ans_gtcrn_onnx" == tag:
            from . import ans_gtcrn_onnx

        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        return engine

    @staticmethod
    def initEngine(tag: str | None = None, **kwargs) -> interface.ISpeechEnhancer | EngineClass:
        # punc
        tag = tag or os.getenv("SPEECH_ENHANCER_TAG", "enhancer_ans_rnnoise")
        logging.info(f"{tag} args: {kwargs}")
        engine = SpeechEnhancerEnvInit.getEngine(tag, **kwargs)
        logging.info(f"initEngine: {tag}, {engine}")
        return engine
