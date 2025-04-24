from dataclasses import dataclass
from typing import Union


@dataclass
class Code2WavGenerationConfig:
    # dit cfm
    num_steps: int = 10
    guidance_scale: float = 0.5
    sway_coefficient: float = -1.0


@dataclass
class Code2WavEngineConfig(Code2WavGenerationConfig):
    model_path: str = ""
    enable_torch_compile: bool = False 
    enable_torch_compile_first_chunk: bool = False
    odeint_method: str = "euler"
    odeint_method_relaxed: bool = False
    batched_chunk: int = 3
    frequency: str = "50hz"
    device: Union[int, str] = "cuda"
    code2wav_dynamic_batch: bool = False
