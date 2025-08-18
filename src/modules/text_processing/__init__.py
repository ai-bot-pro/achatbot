import logging
import os

from src.common import interface
from src.common.factory import EngineClass, EngineFactory


from dotenv import load_dotenv

load_dotenv(override=True)


class TextProcessingEnvInit:
    @staticmethod
    def getEngine(tag, **kwargs) -> interface.ITextProcessing | EngineClass:
        if "we_text_processing" == tag:
            from . import we_text_processing

        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        return engine

    @staticmethod
    def initEngine(tag: str | None = None, **kwargs) -> interface.ITextProcessing | EngineClass:
        # text normalize processing
        tag = tag or os.getenv("TEXT_PROCESSING_TAG", "we_text_processing")
        logging.info(f"{tag} args: {kwargs}")
        engine = TextProcessingEnvInit.getEngine(tag, **kwargs)
        logging.info(f"initEngine: {tag}, {engine}")
        return engine
