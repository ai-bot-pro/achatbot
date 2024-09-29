from dataclasses import dataclass
from typing import List


@dataclass
class VisionDetectorArgs:
    model_path: str = "yolov8n"
    task: str | None = None
    verbose: bool = False
    confidence_score: float = 0.5
    custom_classes: List[str] = []
