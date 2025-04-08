import os
from pydantic import BaseModel

from src.common.types import MODELS_DIR


class CodecArgs(BaseModel):
    model_dir: str = os.path.join(MODELS_DIR, "")
    device: str | None = None

    # for single config file
    config_path: str = os.path.join(MODELS_DIR, "config.yaml")
    model_path: str = os.path.join(MODELS_DIR, "model.ckpt")
