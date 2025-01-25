import os
import logging

from dotenv import load_dotenv

from src.common.factory import EngineClass, EngineFactory
from src.types.codec import CodecArgs
from src.modules.codec.interface import ICodec


load_dotenv(override=True)


class CodecEnvInit:
    @staticmethod
    def getEngine(tag, **kwargs) -> ICodec | EngineClass:
        if "codec_transformers_mimi" == tag:
            from .audio import transformers_mimi
        elif "codec_moshi_mimi" == tag:
            from .audio import moshi_mimi
        elif "codec_xcodec2" == tag:
            from .audio import xcodec2

        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        return engine

    @staticmethod
    def initCodecEngine(tag=None, **kwargs) -> ICodec | EngineClass:
        # codec
        tag = tag or os.getenv("CODEC_TAG", "codec_xcodec2")
        kwargs = kwargs or CodecEnvInit.map_config_func[tag]()
        engine = CodecEnvInit.getEngine(tag, **kwargs)
        logging.info(f"initVADEngine: {tag}, {engine}")
        return engine

    @staticmethod
    def get_args() -> dict:
        res = CodecArgs(model_dir=os.getenv("CODEC_MODEL_DIR", "")).__dict__
        return res

    # TAG : config
    map_config_func = {
        "codec_xcodec2": get_args,
        "codec_transformers_mimi": get_args,
        "codec_moshi_mimi": get_args,
    }
