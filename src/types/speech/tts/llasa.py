import os
from pathlib import Path
from typing_extensions import Annotated
from pydantic import BaseModel, Field, conint

from src.common.types import MODELS_DIR


class FishSpeechTTSInferArgs(BaseModel):
    pass