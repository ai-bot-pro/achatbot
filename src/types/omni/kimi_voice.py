from dataclasses import dataclass, field

from src.types.llm.transformers import TransformersLMArgs


@dataclass
class KimiAudioDeTokenizerArgs:
    device: str = None
    look_ahead_tokens: int = 12
    max_prompt_chunk: int = 10  # 10 * 3 = 30s
    max_kv_cache_tokens: int = 900
    use_cfg: bool = False
    use_cfg_rescale: bool = True
    cfg_init: float = 1.5
    cfg_scale: float = 7.5
    cfg_schedule: str = "linear"  # no use


@dataclass
class KimiAudioTransformersVoiceLMArgs(TransformersLMArgs):
    """
    text+voice(audio+speech) lm args + double heads(text|audio) sampling  + code2wav(Audio-DeTokenizer)(dit cfm + bigvgan vocoder) args
    """

    is_load_detokenizer: bool = True
    code2wav_args: dict = field(default_factory=lambda: KimiAudioDeTokenizerArgs().__dict__)
