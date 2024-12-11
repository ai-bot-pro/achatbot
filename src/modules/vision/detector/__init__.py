import logging
import os

from src.common import interface
from src.common.factory import EngineClass, EngineFactory
from src.common.types import MODELS_DIR
from src.types.vision.detector.yolo import VisionDetectorArgs

from dotenv import load_dotenv

load_dotenv(override=True)


class VisionDetectorEnvInit:
    @staticmethod
    def getEngine(tag, **kwargs) -> interface.IVisionDetector | EngineClass:
        if "vision_yolo_detector" in tag:
            from . import yolo

        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        return engine

    @staticmethod
    def initVisionDetectorEngine(
        tag: str | None = None, kwargs: dict | None = None
    ) -> interface.IVisionDetector | EngineClass:
        # vision detector
        tag = tag if tag else os.getenv("VISION_DETECTOR_TAG", "vision_yolo_detector")
        if kwargs is None:
            kwargs = VisionDetectorEnvInit.map_config_func[tag]()
        engine = VisionDetectorEnvInit.getEngine(tag, **kwargs)
        logging.info(f"initVisionDetectorEngine: {tag}, {engine}")
        return engine

    @staticmethod
    def get_yolo_detector_args() -> dict:
        kwargs = VisionDetectorArgs(
            verbose=bool(os.getenv("YOLO_VERBOSE", "0")),
            stream=bool(os.getenv("YOLO_STREAM", "0")),
            model=os.getenv("YOLO_MODEL", os.path.join(MODELS_DIR, "yolov8n")),
        ).__dict__

        return kwargs

    # TAG : config
    map_config_func = {
        "vision_yolo_detector": get_yolo_detector_args,
    }
