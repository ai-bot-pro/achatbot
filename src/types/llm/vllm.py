from dataclasses import dataclass, field

from vllm import AsyncEngineArgs

from .sampling import LMGenerateArgs


@dataclass
class VllmEngineArgs:
    """
    vllm language model engine args
    """

    serv_args: dict = field(default_factory=lambda: AsyncEngineArgs().__dict__)
    gen_args: dict = field(default_factory=lambda: LMGenerateArgs().__dict__)
