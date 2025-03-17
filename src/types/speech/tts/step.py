from dataclasses import dataclass, field
import os

from src.common.types import MODELS_DIR
from src.types.llm.transformers import TransformersSpeechLMArgs


@dataclass
class StepTTSArgs:
    """
    TransformersManualSpeechStep LM (Step-1) + Linguistic and Semantic Tokenizer(ref audio encoder) -> Step-Audio TTS
    """

    lm_args: dict = field(default_factory=lambda: TransformersSpeechLMArgs().__dict__)
    # >=2 increase for better speech quality, but rtf slow (speech quality vs rtf)
    stream_factor: int = 2

    tts_mode: str = "lm_gen"  # lm_gen(lm_gen->flow->hifi), voice_clone(no lm_gen, flow->hifi)

    # Linguistic and Semantic speech Tokenizer(ref audio encoder) args
    speech_tokenizer_model_path: str = os.path.join(MODELS_DIR, "stepfun-ai/Step-Audio-Tokenizer")
