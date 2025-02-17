from dataclasses import dataclass, field
import os

from src.common.types import MODELS_DIR
from src.types.codec import CodecArgs


@dataclass
class ZonosTTSArgs:
    """
    Transformers/Mabama2 LM + codec(DAC) -> zonos TTS
    """

    lm_checkpoint_dir: str = str(os.path.join(MODELS_DIR, "Zyphra/Zonos-v0.1-transformer"))

    codec_args: dict = field(default_factory=CodecArgs().__dict__)

    # ref audio file path and prompt text
    ref_audio_file_path: str = ""
    prompt_text: str = ""
    # is save ref audio encode vq code indices .npy
    is_save: bool = False
    output_codebook_indices_dir: str = os.path.join(MODELS_DIR, "zonos_codebook_indices")

    # random seed
    seed: int = 42

    # streaming generation chunk size
    # less for faster streaming but lower quality
    chunk_size: int = 40

    # warm up
    warmup_steps: int = 1
    warm_up_text: str = "hello world"

    # sample conditioning params
    language: str = "en-us"
    emotion: list[float] = [0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077]
    fmax: float = 22050.0
    pitch_std: float = 20.0
    speaking_rate: float = 15.0
    vqscore_8: list[float] = [0.78] * 8
    ctc_loss: float = 0.0
    dnsmos_ovrl: float = 4.0

    # streaming
    tts_stream: bool = True
    chunk_length_seconds: float = 0.5

    # silence
    add_silence_chunk: bool = False
