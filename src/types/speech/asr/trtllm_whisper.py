from dataclasses import dataclass
import os

from src.types.speech.asr.base import ASRArgs
from src.common.types import ASSETS_DIR, MODELS_DIR


@dataclass
class WhisperTensorRTLLMASRArgs(ASRArgs):
    use_py_session: bool = True
    log_level: str = "info"

    engine_dir: str = os.path.join(MODELS_DIR, "whisper", "trt_engines_float16")
    assets_dir: str = ASSETS_DIR
    max_input_len: int = 3000
    max_output_len: int = 96
    max_batch_size: int = 1  # need <= build decoder_model_config.max_batch_size
    kv_cache_free_gpu_memory_fraction: float = 0.9
    cross_kv_cache_fraction: float = 0.5
    num_beams: int = 1
    debug: bool = False

    text_prefix: str = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
    """
    1. max: pad to the 30s, using the option if the model is trained with max padding e.g. openai official models,
    2. longest: pad to the longest sequence in the batch,
    3. nopad: no padding, only works with cpp session,
    """
    padding_strategy: str = "max"
