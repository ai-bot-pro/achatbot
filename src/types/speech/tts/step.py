from dataclasses import dataclass, field
import os

from src.common.types import MODELS_DIR
from src.types.llm.transformers import TransformersSpeechLMArgs


@dataclass
class StepTTSArgs:
    """
    TransformersManualSpeechLlasa LM + Linguistic and Semantic Tokenizer(ref audio encoder) -> llasa TTS
    """

    lm_args: dict = field(default_factory=TransformersSpeechLMArgs().__dict__)
    # >=2 increase for better speech quality, but rtf slow (speech quality vs rtf)
    stream_factor: int = 2

    # Linguistic and Semantic Tokenizer(ref audio encoder) args
    tokenizer_model_path: str = os.path.join(MODELS_DIR, "stepfun-ai/Step-Audio-Tokenizer")

    # ref audio file path and prompt text
    ref_audio_file_path: str = ""
    prompt_text: str = ""
    # is save ref audio encode vq code indices .npy
    is_save: bool = False
    output_codebook_indices_dir: str = os.path.join(MODELS_DIR, "step_codebook_indices")
