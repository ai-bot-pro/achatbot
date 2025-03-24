from dataclasses import dataclass, field
import os

from src.common.types import MODELS_DIR
from src.types.codec import CodecArgs
from src.types.llm.transformers import TransformersSpeechLMArgs


@dataclass
class OrpheusTTSArgs:
    """
    TransformersManualSpeech LM (llama3) + SNAC ->  TTS
    """

    lm_args: dict = field(default_factory=lambda: TransformersSpeechLMArgs().__dict__)
    codec_args: dict = field(default_factory=lambda: CodecArgs().__dict__)
    voice_name: str = "tara"

    # static batch
    stream_factor: int = 1  # 1*chunk_size(28)
    token_overlap_len: int = 0

    # stream
    tts_stream: bool = True
    chunk_length_seconds: int = 1

    # silence
    add_silence_chunk: bool = False
