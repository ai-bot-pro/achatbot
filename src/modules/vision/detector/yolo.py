import logging
from typing import Any, Callable, Dict

from PIL import Image
import numpy as np

try:
    from ultralytics import YOLO
    import supervision as sv
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use yolo model, you need to `pip install achatbot[vision_yolo_detector]`"
    )
    raise Exception(f"Missing module: {e}")

from src.common.device_cuda import CUDAInfo
from src.types.vision.detector.yolo import VisionDetectorArgs
from src.common.factory import EngineClass
from src.common.interface import IVisionDetector


class VisionYoloDetector(EngineClass, IVisionDetector):
    """
    You Only Look Once
    see:
    - https://docs.ultralytics.com/models/
    - https://supervision.roboflow.com/develop/how_to/detect_and_annotate/
    """

    TAG = "vision_yolo_detector"

    def __init__(self, **args) -> None:
        self.args = VisionDetectorArgs(**args)
        self._detected_filter: Callable[[sv.Detections], sv.Detections] = None
        device = "cpu"
        if self.args.device:
            device = self.args.device
        else:
            info = CUDAInfo()
            if info.is_cuda:
                device = "cuda"
        self._model = YOLO(self.args.model, task=self.args.task, verbose=self.args.verbose).to(
            device
        )
        if hasattr(self._model, "set_classes"):
            # for yolo-world
            if len(self.args.custom_classes) > 0:
                self._model.set_classes(self.args.custom_classes)
        self._class_names = list(self._model.names.values())

        model_million_params = sum(p.numel() for p in self._model.parameters()) / 1e6
        logging.info(f"{self.TAG} have {model_million_params}M parameters on {self._model.device}")
        logging.info(f"class_names:{self._class_names}")
        logging.debug(self._model)

    @property
    def class_names(self):
        return self._class_names

    def detect(self, session) -> bool:
        """
        TODO: cpu/gpu binding, need optimize to 10ms< with gpu acceleration
        """
        if "detect_img" not in session.ctx.state:
            logging.warning("No detect_img in session ctx state")
            return False

        kwargs = {}
        if self.args.selected_classes:
            selected_ind = [
                self._class_names.index(option) for option in self.args.selected_classes
            ]
            if not isinstance(selected_ind, list):
                selected_ind = list(selected_ind)
            kwargs["classes"] = selected_ind
        if self.args.conf is not None:
            kwargs["conf"] = self.args.conf
        if self.args.iou is not None:
            kwargs["iou"] = self.args.iou
        # print("kwargs--->:", kwargs)
        results = self._model.predict(
            session.ctx.state["detect_img"],
            stream=self.args.stream,
            **kwargs,
        )
        cf_dict = {}
        for item in results:
            detections = sv.Detections.from_ultralytics(item)
            cf_dict = {}
            for name in detections.data["class_name"]:
                cf_dict[name] = {"d_cn": 0, "d_cf": []}
            for t in zip(
                detections.xyxy,
                detections.confidence,
                detections.class_id,
                detections.data["class_name"],
            ):
                cf_dict[t[3]]["d_cn"] += 1
                cf_dict[t[3]]["d_cf"].append(t[1])

            eval_str = ""
            for item in self.args.custom_confidences:  # order to check
                if item.class_name in cf_dict:
                    op = item.boolean_op.lower()
                    is_detected = (
                        max(cf_dict[item.class_name]["d_cf"]) > item.d_confidence
                        and cf_dict[item.class_name]["d_cn"] >= item.d_min_cn
                    )
                    logging.debug(f"item:{item}, cf_dict:{cf_dict}, is_detcted:{is_detected}")
                    if op == "or":
                        eval_str = f"{eval_str} or {is_detected}"
                    else:
                        eval_str = f"{eval_str} and {is_detected}"
            eval_str = eval_str.lstrip(" or").lstrip(" and")
            logging.debug(f"eval_str:{eval_str}")
            if eval_str and eval(eval_str):
                return True

        return False

    def set_filter(self, hanlder: Callable[[sv.Detections], sv.Detections]):
        self._detected_filter = hanlder

    def annotate(self, session):
        """
        TODO: need optimize to 10ms< with gpu acceleration
        """
        if "detect_img" not in session.ctx.state:
            logging.warning("No detect_img in session ctx state")
            yield None
            return

        img = session.ctx.state["detect_img"]
        results = self._model(img, stream=self.args.stream)
        for item in results:
            detections = sv.Detections.from_ultralytics(item)
            if self._detected_filter:
                detections = self._detected_filter(detections)

            # !TODO: Annotator Args to use for different annotators @weedge
            match self.args.annotator_type:
                case "box":
                    annotator = sv.BoxAnnotator()
                case "mask":
                    annotator = sv.MaskAnnotator()
                case _:
                    annotator = sv.BoxAnnotator()

            label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)
            labels = [
                f"{class_name} {confidence:.2f}"
                for class_name, confidence in zip(detections["class_name"], detections.confidence)
            ]

            annotated_img = annotator.annotate(scene=img, detections=detections)
            annotated_img = label_annotator.annotate(
                scene=annotated_img, detections=detections, labels=labels
            )
            if isinstance(annotated_img, np.ndarray):
                annotated_img = Image.fromarray(annotated_img)

            yield annotated_img
