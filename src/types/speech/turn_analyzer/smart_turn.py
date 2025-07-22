import os
from pydantic import BaseModel, Field

from src.common.types import MODELS_DIR, RATE


# Default timing parameters
STOP_SECS = 3
PRE_SPEECH_MS = 0
MAX_DURATION_SECONDS = 8  # Max allowed segment duration


class SmartTurnArgs(BaseModel):
    """Configuration parameters for smart turn analysis.

    Parameters:
        stop_secs: Maximum silence duration in seconds before ending turn.
        pre_speech_ms: Milliseconds of audio to include before speech starts.
        max_duration_secs: Maximum duration in seconds for audio segments.
    """

    model_path: str = os.path.join(MODELS_DIR, "pipecat-ai/smart-turn-v2")
    torch_dtype: str = Field(
        default="float32",
        description="The PyTorch data type for the model and input tensors. One of `float32` (full-precision),  `bfloat16` (half-precision), default float32.",
    )
    warmup_steps: int = 2
    sample_rate: int = RATE
    stop_secs: float = STOP_SECS
    pre_speech_ms: float = PRE_SPEECH_MS
    max_duration_secs: float = MAX_DURATION_SECONDS
