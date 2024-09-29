import logging

from types.vision import VisionDetectorArgs

try:
    from ultralytics import YOLO
    import supervision as sv
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        f"In order to use yolo model, you need to `pip install achatbot[vision_yolo_detector]`")
    raise Exception(f"Missing module: {e}")

from src.common.factory import EngineClass
from src.common.interface import IVisionDetector


class VisionYoloDetector(EngineClass, IVisionDetector):
    TAG = "vision_yolo_detector"

    def __init__(self, **args) -> None:
        self.args = VisionDetectorArgs(**args)
        self._model = YOLO(self.args.model_path,
                           task=self.args.task, verbose=self.args.verbose)

    def detect(self, session) -> bool:
        if "detect_img" not in session.ctx.state:
            logging.warning("No detect_img in session ctx state")
            return

        results = self._model(session.ctx.state['detect_img'])[0]
        detections = sv.Detections.from_ultralytics(results)
        return detections
