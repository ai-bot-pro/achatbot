import logging


try:
    from ultralytics import YOLO
    import supervision as sv
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        f"In order to use yolo model, you need to `pip install achatbot[vision_yolo_detector]`")
    raise Exception(f"Missing module: {e}")

from src.types.vision import VisionDetectorArgs
from src.common.factory import EngineClass
from src.common.interface import IVisionDetector


class VisionYoloDetector(EngineClass, IVisionDetector):
    """
    You Only Look Once
    """
    TAG = "vision_yolo_detector"

    def __init__(self, **args) -> None:
        self.args = VisionDetectorArgs(**args)
        self._model = YOLO(self.args.model_path,
                           task=self.args.task,
                           verbose=self.args.verbose)
        if hasattr(self._model, "set_classes"):
            # for yolo-world
            if len(self.args.custom_classes) > 0:
                self._model.set_classes(self.args.custom_classes)
        model_million_params = sum(p.numel() for p in self._model.parameters()) / 1e6
        logging.debug(f"{self.TAG} have {model_million_params}M parameters")
        logging.debug(self._model)

    def detect(self, session) -> bool:
        if "detect_img" not in session.ctx.state:
            logging.warning("No detect_img in session ctx state")
            return

        results = self._model(session.ctx.state['detect_img'], stream=self.args.stream)
        cf_dict = {}
        for item in results:
            detections = sv.Detections.from_ultralytics(item)
            cf_dict = {}
            for name in detections.data['class_name']:
                cf_dict[name] = {"d_cn": 0, "d_cf": []}
            for t in zip(
                    detections.xyxy,
                    detections.confidence,
                    detections.class_id,
                    detections.data['class_name']):
                cf_dict[t[3]]["d_cn"] += 1
                cf_dict[t[3]]["d_cf"].append(t[1])

            eval_str = ""
            for item in self.args.custom_confidences:  # order to check
                if item.class_name in cf_dict:
                    op = item.boolean_op.lower()
                    is_detected = max(cf_dict[item.class_name]["d_cf"]) > item.d_confidence \
                        and cf_dict[item.class_name]["d_cn"] >= item.d_min_cn
                    logging.debug(f"item:{item}, cf_dict:{cf_dict}, is_detcted:{is_detected}")
                    if op == "or":
                        eval_str = f"{eval_str} or {is_detected}"
                    else:
                        eval_str = f"{eval_str} and {is_detected}"
            eval_str = eval_str.lstrip(" or").lstrip(" and")
            logging.debug(f"eval_str:{eval_str}")
            if eval(eval_str):
                return True

        return False
