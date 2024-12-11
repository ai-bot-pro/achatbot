import logging
import os

from src.common import interface
from src.common.factory import EngineClass, EngineFactory
from src.common.types import MODELS_DIR
from src.types.vision.ocr.transformers_got import TransformersGoTOCRArgs

from dotenv import load_dotenv

load_dotenv(override=True)


class VisionOCREnvInit:
    @staticmethod
    def getEngine(tag, **kwargs) -> interface.IVisionOCR | EngineClass:
        if "vision_transformers_got_ocr" in tag:
            from . import transformers_got

        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        return engine

    @staticmethod
    def initVisionOCREngine(
        tag: str | None = None, kwargs: dict | None = None
    ) -> interface.IVisionOCR | EngineClass:
        # vision OCR
        tag = tag if tag else os.getenv("VISION_OCR_TAG", "vision_transformers_got_ocr")
        if kwargs is None:
            kwargs = VisionOCREnvInit.map_config_func[tag]()
        engine = VisionOCREnvInit.getEngine(tag, **kwargs)
        logging.info(f"initVisionOCREngine: {tag}, {engine}")
        return engine

    @staticmethod
    def get_transformers_got_ocr_args() -> dict:
        kwargs = TransformersGoTOCRArgs(
            lm_model_name_or_path=os.getenv(
                "LM_GOT_OCR_MODEL", os.path.join(MODELS_DIR, "stepfun-ai/GOT-OCR2_0")
            ),
            lm_attn_impl=os.getenv("LM_ATTN_IMPL", None),
            lm_device_map=os.getenv("LM_DEVICE_MAP", ""),
            lm_device=os.getenv("LM_DEVICE", "cpu"),
            lm_torch_dtype=os.getenv("LM_TORCH_DTYPE", "bfloat16"),
            lm_stream=bool(os.getenv("LM_STREAM", "1")),
            model_type=os.getenv("GOT_MODEL_TYPE", "chat"),
            ocr_type=os.getenv("GOT_OCR_TYPE", "ocr"),
            ocr_box=os.getenv("GOT_OCR_BOX", ""),
            ocr_color=os.getenv("GOT_OCR_COLOR", ""),
        ).__dict__

        return kwargs

    # TAG : config
    map_config_func = {
        "vision_transformers_got_ocr": get_transformers_got_ocr_args,
    }
