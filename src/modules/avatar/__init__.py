import os
import logging

from dotenv import load_dotenv

from src.common.types import MODELS_DIR
from src.common.factory import EngineClass, EngineFactory
from src.modules.avatar.interface import IFaceAvatar


load_dotenv(override=True)


class AvatarEnvInit:
    @staticmethod
    def getEngine(tag, **kwargs) -> IFaceAvatar | EngineClass:
        if "lite_avatar" == tag:
            from . import lite_avatar

        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        return engine

    @staticmethod
    def initFaceAvatarEngine(tag=None, **kwargs) -> IFaceAvatar | EngineClass:
        # Avatar
        tag = tag or os.getenv("FACE_AVATAR_TAG", "lite_avatar")
        kwargs = kwargs or AvatarEnvInit.map_config_func[tag]()
        engine = AvatarEnvInit.getEngine(tag, **kwargs)
        logging.info(f"initAvatarEngine: {tag}, {engine}")
        return engine

    @staticmethod
    def get_lite_args() -> dict:
        from src.types.avatar.lite_avatar import AvatarInitOption

        res = AvatarInitOption(
            weight_dir=os.getenv(
                "AVATAR_MODEL_DIR", os.path.join(MODELS_DIR, "weege007/liteavatar")
            )
        ).__dict__
        return res

    # TAG : config
    map_config_func = {
        "lite_avatar": get_lite_args,
    }
