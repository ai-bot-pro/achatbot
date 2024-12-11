from typing import Any, Dict, List, Optional

from pydantic import BaseModel, PrivateAttr


class CustomConfidence(BaseModel):
    """
    # just for one level check,from right to left check, e.g:
    a,b,c = True,False,False
    a and b and c => False
    a or b and c => True
    a and b or c => False
    a or b or c => True
    """

    boolean_op: str = "and"  # `is_detected` | and `is_detected` | or `is_detected`
    class_name: str = "person"
    d_min_cn: int = 1  # min detected obj cn in image
    d_confidence: float = 0.25  # beyond Confidence Threshold
    # d_iou: float = 0.45  # beyond IoU Threshold


class VisionDetectorArgs(BaseModel):
    model: str = "yolov8n"
    device: Optional[str] = None
    task: Optional[str] = None
    verbose: bool = False
    stream: bool = False
    conf: Optional[float] = None  # confidence
    iou: Optional[float] = None  # IoU
    selected_classes: Optional[List[str]] = None
    custom_classes: List[str] = ["person"]
    custom_confidences: List[CustomConfidence] = [CustomConfidence()]
    _custom_confidences_dict: Dict[str, CustomConfidence] = PrivateAttr(default={})
    annotator_type: str = "box"  # default box, box | mask
    desc: str | list = ""  # (in) detected describe by custom
    out_desc: str | list = ""  # un(out) detected describe by custom

    def model_post_init(self, __context: Any) -> None:
        self._custom_confidences_dict = {}
        for conf in self.custom_confidences:
            self._custom_confidences_dict[conf.class_name] = conf
        return super().model_post_init(__context)
