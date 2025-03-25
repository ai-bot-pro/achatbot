from dataclasses import dataclass, field


from .sampling import LMGenerateArgs


@dataclass
class TensorRTLLMEngineArgs:
    """
    TensorRT-llm language model engine args
    """

    serv_args: dict = {}
    gen_args: dict = field(default_factory=lambda: LMGenerateArgs().__dict__)
