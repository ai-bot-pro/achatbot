from dataclasses import dataclass, field

from src.types.llm.transformers import TransformersLMArgs


@dataclass
class VitaAudioTokenizerArgs:
    audio_tokenizer_type: str = "sensevoice_glm4voice"  # glm4voice | sensevoice_glm4voice
    audio_tokenizer_model_path: str = None
    sense_voice_model_path: str = None
    flow_path: str = None
    audio_tokenizer_rank: int = 0


@dataclass
class VitaAudioTransformersVoiceLMArgs(TransformersLMArgs, VitaAudioTokenizerArgs):
    """ """
