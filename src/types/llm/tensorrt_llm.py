from dataclasses import dataclass, field

from tensorrt_llm.llmapi.llm_utils import LlmArgs


from .sampling import LMGenerateArgs


@dataclass
class TensorRTLLMEngineArgs:
    """
    TensorRT-llm language model engine args
    """

    serv_args: dict = field(default_factory=lambda: LlmArgs().__dict__)
    gen_args: dict = field(default_factory=lambda: LMGenerateArgs().__dict__)
