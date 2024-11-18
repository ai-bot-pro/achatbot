from dataclasses import dataclass, field


@dataclass
class LMGenArgs:
    """moshi lm generation defualt arguments"""
    use_sampling: bool = True
    temp: float = 0.8
    temp_text: float = 0.7
    top_k: int = 250
    top_k_text: int = 25
    check: bool = False
