from dataclasses import dataclass, field
import os

from src.common.types import MODELS_DIR


@dataclass
class VibeVoiceTTSArgs:
    """
    qwen2.5 + difusion -> vibevoice TTS
    """

    device: str | None = None
    model_path: str = os.path.join(MODELS_DIR, "microsoft/VibeVoice-Realtime-0.5B")
    speaker_embedding_pt_dir: str = os.path.join(
        MODELS_DIR, "microsoft/VibeVoice-Realtime-0.5B/voices/streaming_model"
    )

    # generate params
    cfg_scale: float = 1.5
    inference_steps: int = 5
    voice: str | None = None
    do_sample: bool = False
    temperature: float = 0.9
    top_p: float = 0.9
    refresh_negative: bool = True

    # random seed
    seed: int = 42

    # warm up
    warmup_steps: int = 1
    warm_up_text: str = "hello world"

    # streaming
    tts_stream: bool = True
    chunk_length_seconds: float = 0.5

    # silence
    add_silence_chunk: bool = False
