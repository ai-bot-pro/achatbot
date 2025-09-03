from dataclasses import dataclass, field

from sglang.srt.server_args import ServerArgs
from .sampling import LMGenerateArgs


@dataclass
class SGLangEngineArgs:
    """
    SGLang language model engine args
    """

    serv_args: dict = field(default_factory=lambda: ServerArgs().__dict__)
    gen_args: dict = field(default_factory=lambda: LMGenerateArgs().__dict__)
    warmup_prompt: str = "hello"
    warmup_steps: int = 0