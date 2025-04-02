from dataclasses import dataclass, field
from typing import List, Optional


from tensorrt_llm.llmapi.llm_utils import LlmArgs


from .sampling import LMGenerateArgs


@dataclass
class TensorRTLLMEngineArgs:
    """
    TensorRT-llm language model engine args
    """

    serv_args: dict = field(default_factory=lambda: LlmArgs().__dict__)
    gen_args: dict = field(default_factory=lambda: LMGenerateArgs().__dict__)


@dataclass
class TensorRTLLMRunnerArgs:
    """
    TensorRT-llm language model runner args
    """

    engine_dir: str
    max_output_len: Optional[int] = None
    lora_dir: Optional[List[str]] = None
    rank: int = 0
    debug_mode: bool = False
    lora_ckpt_source: str = "hf"
    medusa_choices: List[List[int]] = None
    # stream: torch.cuda.Stream = None
    gpu_weights_percent: float = 1
    enable_context_fmha_fp32_acc: Optional[bool] = None
    multi_block_mode: Optional[bool] = None


@dataclass
class TensorRTLLMRunnerEngineArgs:
    """
    TensorRT-llm language model runner engine args
    """

    serv_args: dict = field(default_factory=lambda: TensorRTLLMRunnerArgs().__dict__)
    gen_args: dict = field(default_factory=lambda: LMGenerateArgs().__dict__)
