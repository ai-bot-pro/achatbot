from dataclasses import dataclass, field
import os

from src.common.types import MODELS_DIR
from src.types.llm.transformers import TransformersSpeechLMArgs, TransformersLMArgs


@dataclass
class SparkTTSArgs:
    """
    TransformersManualSpeechSpark LM (qwen2.5) + Global and Semantic Tokenizer(ref audio encoder) -> Spark-Audio TTS
    """

    # "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device: str = None

    # LLM + BiCodecTokenizer + Wav2Vec2(feature exactor)
    model_dir: str = os.path.join(MODELS_DIR, "SparkAudio/Spark-TTS-0.5B")

    lm_args: dict = field(default_factory=lambda: TransformersSpeechLMArgs().__dict__)

    # defualt static batch
    stream_factor: int = 2
    stream_scale_factor: float = 1.0
    max_stream_factor: int = 2
    token_overlap_len: int = 0
    input_frame_rate: int = 25

    ref_text: str = ""
    ref_audio_path: str = ""

    # stream
    tts_stream: bool = True
    chunk_length_seconds: int = 1

    # silence
    add_silence_chunk: bool = False


@dataclass
class SparkGeneratorTTSArgs(SparkTTSArgs):
    """
    Spark LM (qwen2.5) Generator + LM Tokenizer + Global and Semantic Tokenizer(ref audio encoder) -> Spark-Audio TTS
    """

    lm_generator_tag: str = "llm_transformers_generator"
    lm_args: dict = field(default_factory=lambda: TransformersLMArgs().__dict__)

    lm_tokenzier_dir: str = os.path.join(MODELS_DIR, "SparkAudio/Spark-TTS-0.5B/LLM")
