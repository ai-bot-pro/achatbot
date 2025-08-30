import logging
import os

from src.common import interface
from src.common.factory import EngineClass, EngineFactory


from dotenv import load_dotenv

load_dotenv(override=True)


class PuncEnvInit:
    @staticmethod
    def getEngine(tag, **kwargs) -> interface.IPunc | EngineClass:
        if "punc_ct_tranformer" == tag:
            from . import ct_transformer
        elif "punc_ct_tranformer_offline" == tag:
            from . import ct_transformer
        elif "punc_ct_tranformer_onnx" == tag:
            from . import ct_transformer_onnx
        elif "punc_ct_tranformer_onnx_offline" == tag:
            from . import ct_transformer_onnx

        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        return engine

    @staticmethod
    def initEngine(tag: str | None = None, **kwargs) -> interface.IPunc | EngineClass:
        # punc
        tag = tag or os.getenv("PUNC_TAG", "punc_ct_tranformer")
        logging.info(f"{tag} args: {kwargs}")
        engine = PuncEnvInit.getEngine(tag, **kwargs)
        logging.info(f"initEngine: {tag}, {engine}")
        return engine
