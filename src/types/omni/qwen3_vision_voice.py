from dataclasses import dataclass, field

from src.types.llm.transformers import TransformersLMArgs


@dataclass
class Qwen3OmniCode2WavArgs:
    """Streaming Codec Decoder"""

    chunk_size: int = 50
    left_context_size: int = 25


@dataclass
class Qwen3TransformersVisionVoiceLMArgs(TransformersLMArgs):
    """
    text+vision(Image/video)+voice(audio+speech) lm args (thinker + talker) + code2wav(Streaming Codec Decoder) args
    """

    thinker_eos_token_ids: list = field(default_factory=lambda: [151643, 151645])
    thinker_stop_strings_per_step: list = field(default_factory=lambda: [".", "。"])
    thinker_args: dict = field(default_factory=lambda: TransformersLMArgs().__dict__)
    talker_args: dict = field(default_factory=lambda: TransformersLMArgs().__dict__)
    talker_skip_thinker_token_ids: list[int] = field(default_factory=lambda: [])
    talker_eos_token_ids: list[int] = field(default_factory=lambda: [8292, 8294])
    speaker: str = "Chelsie"
    disable_talker: bool = False
    code2wav_args: dict = field(default_factory=lambda: Qwen3OmniCode2WavArgs().__dict__)
