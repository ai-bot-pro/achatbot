import os
from pydantic import BaseModel

from src.common.types import MODELS_DIR


class CodecArgs(BaseModel):
    model_dir: str = os.path.join(MODELS_DIR, "")
    device: str | None = None
