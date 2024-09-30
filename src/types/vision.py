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
    d_min_cn: int = 1
    d_confidence: float = 0.5


class VisionDetectorArgs(BaseModel):
    model: str = "yolov8n"
    task: Optional[str] = None
    verbose: bool = False
    stream: bool = False
    custom_classes: List[str] = ["person"]
    custom_confidences: List[CustomConfidence] = [CustomConfidence()]
    _custom_confidences_dict: Dict[str, CustomConfidence] = PrivateAttr(default={})
    annotator_type: str = "box"  # default box, box | mask

    def model_post_init(self, __context: Any) -> None:
        self._custom_confidences_dict = {}
        for conf in self.custom_confidences:
            self._custom_confidences_dict[conf.class_name] = conf
        return super().model_post_init(__context)
