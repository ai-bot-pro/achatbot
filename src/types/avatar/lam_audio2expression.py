import os
from src.common.types import MODELS_DIR, RATE
from pydantic import BaseModel


class LAMAudio2ExpressionAvatarArgs(BaseModel):
    weight_path: str = os.path.join(
        MODELS_DIR, "LAM_audio2exp/pretrained_models/lam_audio2exp_streaming.tar"
    )
    wav2vec_dir: str = os.path.join(MODELS_DIR, "facebook/wav2vec2-base-960h")
    audio_sample_rate: int = RATE
