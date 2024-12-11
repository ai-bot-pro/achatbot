from dataclasses import dataclass
from typing import Optional, Union

from src.types.speech.asr.base import ASRArgs


@dataclass
class WhisperTimestampedASRArgs(ASRArgs):
    pass


@dataclass
class WhisperMLXASRArgs(ASRArgs):
    pass


@dataclass
class WhisperTransformersASRArgs(ASRArgs):
    pass


@dataclass
class WhisperGroqASRArgs(ASRArgs):
    timeout_s: float = None
    """
    temperature: The sampling temperature, between 0 and 1. Higher values like 0.8 will make the
    output more random, while lower values like 0.2 will make it more focused and
    deterministic. If set to 0, the model will use
    [log probability](https://en.wikipedia.org/wiki/Log_probability) to
    automatically increase the temperature until certain thresholds are hit.
    """
    temperature: float = 0.0
    # timestamp_granularities: list = []
    # response_format: str = "json"  # json, verbose_json, text
