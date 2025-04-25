from dataclasses import dataclass, field

from src.types.llm.transformers import TransformersLMArgs
from src.thirdparty.qwen2_code2wav import Code2WavEngineConfig


@dataclass
class Qwen2_5TransformersVisionVoiceLMArgs(TransformersLMArgs):
    """
    text+vision(Image/video)+voice(audio+speech) lm args + token2wav(dit cfm + vocoder) args
    """

    thinker_eos_token_ids: list = field(default_factory=lambda: [151644, 151645])
    thinker_stop_strings_per_step: list = field(default_factory=lambda: [".", "。"])
    thinker_args: dict = field(default_factory=lambda: TransformersLMArgs().__dict__)
    talker_args: dict = field(default_factory=lambda: TransformersLMArgs().__dict__)
    talker_skip_thinker_token_ids: list[int] = field(default_factory=lambda: [])
    talker_eos_token_ids: list[int] = field(default_factory=lambda: [8292, 8294])
    code2wav_args: dict = field(default_factory=lambda: Code2WavEngineConfig().__dict__)
    speaker: str = "Chelsie"
    is_use_sliding_window_code2wav: bool = True
    save_wav: bool = False
    disable_talker: bool = False
    thinker_all_talker_stream: bool = False
    mask_embedding: bool = True
