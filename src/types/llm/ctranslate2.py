from dataclasses import dataclass, field
from .sampling import LMGenerateArgs


@dataclass
class Ctranslate2ModelArgs:
    """
    https://opennmt.net/CTranslate2/python/ctranslate2.Generator.html
    """

    model_path: str = ""  # Path to the CTranslate2 model directory.
    device: str = ""  # Device to use (possible values are: cpu, cuda, auto).
    device_index: int = 0  # Device IDs where to place this generator on.
    compute_type: str = "default"  # Model computation type or a dictionary mapping a device name to the computation type (possible values are: default, auto, int8, int8_float32, int8_float16, int8_bfloat16, int16, float16, bfloat16, float32).
    inter_threads: int = 1  # Maximum number of parallel generations.
    intra_threads: int = 0  # Number of OpenMP threads per generator (0 to use a default value).
    max_queued_batches: int = 0  # Maximum numbers of batches in the queue (-1 for unlimited, 0 for an automatic value). When the queue is full, future requests will block until a free slot is available.
    flash_attention: bool = False  # run model with flash attention 2 for self-attention layer,NOTE: need Install from sources and cmake with WITH_FLASH_ATTN
    tensor_parallel: bool = False  # run model with tensor parallel mode.


@dataclass
class Ctranslate2EngineArgs:
    """
    ctranslate2 language model engine args
    """

    model_args: dict = field(default_factory=lambda: Ctranslate2ModelArgs().__dict__)
    gen_args: dict = field(default_factory=lambda: LMGenerateArgs().__dict__)
