
from dataclasses import dataclass
from typing import Optional, Union

from src.types.speech.asr.base import ASRArgs


@dataclass
class WhisperFasterASRArgs(ASRArgs):
    from faster_whisper.vad import VadOptions
    vad_filter: bool = False
    vad_parameters: Optional[Union[dict, VadOptions]] = None
