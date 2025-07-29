from dataclasses import dataclass, field
import os

from src.common.types import ASSETS_DIR, MODELS_DIR
from src.types.llm.transformers import TransformersSpeechLMArgs


@dataclass
class HiggsTTSArgs:
    """
    HiggsAudioModel LM (Llama3.2+HiggsAudioDualFFNDecoderLayer) + HiggsAudioTokenizer (ref audio encoder) -> Higgs-Audio TTS
    """

    # "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device: str = None

    lm_args: dict = field(default_factory=lambda: TransformersSpeechLMArgs().__dict__)
    audio_tokenizer_path: str = os.path.join(MODELS_DIR, "bosonai/higgs-audio-v2-tokenizer")

    # default reference audio
    ref_text: str = "对，这就是我，万人敬仰的太乙真人。"
    ref_audio_path: str = os.path.join(ASSETS_DIR, "basic_ref_zh.wav")

    # stream
    chunk_size: int = 16
    tts_stream: bool = True
    chunk_length_seconds: int = 1

    # silence
    add_silence_chunk: bool = False
