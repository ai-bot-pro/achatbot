from dataclasses import dataclass, field
import os

from src.common.types import MODELS_DIR
from src.types.llm.transformers import TransformersSpeechLMArgs


@dataclass
class SparkTTSArgs:
    """
    TransformersManualSpeechSpark LM (qwen2.5) + Global and Semantic Tokenizer(ref audio encoder) -> Spark-Audio TTS
    """

    # LLM + BiCodecTokenizer + Wav2Vec2(feature exactor)
    model_dir: str = os.path.join(MODELS_DIR, "SparkAudio/Spark-TTS-0.5B")

    lm_args: dict = field(default_factory=TransformersSpeechLMArgs().__dict__)

    # defualt static batch
    stream_factor: int = 2
    stream_scale_factor: float = 1.0
    max_stream_factor: int = 2
    token_overlap_len: int = 0
    input_frame_rate: int = 25

    ref_text: str = ""
    ref_audio_path: str = ""
