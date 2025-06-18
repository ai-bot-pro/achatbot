from enum import Enum
import os
from typing import Any, Optional, TypeVar
from pydantic import BaseModel

import av

from src.common.types import MODELS_DIR, RATE

VIDEO_FPS = 25


class AvatarInitOption(BaseModel):
    audio_sample_rate: int = RATE
    video_frame_rate: int = VIDEO_FPS
    avatar_name: str = "20250408/sample_data"
    is_show_video_debug_text: bool = False
    enable_fast_mode: bool = False
    use_gpu: bool = False
    weight_dir: str = os.path.join(MODELS_DIR, "weege007/liteavatar")
    is_flip: bool = True


class AvatarConfig(BaseModel):
    input_audio_sample_rate: int
    input_audio_slice_duration: float  # input audio duration in second


class SpeechAudio(BaseModel):
    """
    only support mono audio for now
    """

    end_of_speech: bool = False
    speech_id: Any = ""
    sample_rate: int = 16000
    audio_data: bytes = bytes()

    def get_audio_duration(self):
        return len(self.audio_data) / self.sample_rate / 2


class AudioSlice(BaseModel):
    speech_id: Any
    play_audio_data: bytes
    play_audio_sample_rate: int
    algo_audio_data: Optional[bytes]
    algo_audio_sample_rate: int
    end_of_speech: bool
    front_padding_duration: float = 0
    end_padding_duration: float = 0

    def get_audio_duration(self) -> float:
        return len(self.play_audio_data) / self.play_audio_sample_rate / 2


class AvatarStatus(Enum):
    SPEAKING = 0
    LISTENING = 1


SignalType = TypeVar("SignalType")


class SignalResult(BaseModel):
    speech_id: Any
    end_of_speech: bool
    avatar_status: AvatarStatus
    audio_slice: Optional[AudioSlice] = None
    frame_id: int
    global_frame_id: int = 0
    middle_data: SignalType


class MouthResult(BaseModel):
    speech_id: Any
    avatar_status: AvatarStatus
    end_of_speech: bool
    bg_frame_id: int
    mouth_image: Any
    audio_slice: Optional[AudioSlice] = None
    global_frame_id: int

    model_config = {"arbitrary_types_allowed": True}


class VideoResult(BaseModel):
    speech_id: Any
    avatar_status: AvatarStatus
    video_frame: Any | av.VideoFrame
    end_of_speech: bool

    model_config = {"arbitrary_types_allowed": True}


class AudioResult(BaseModel):
    speech_id: Any
    audio_frame: bytes | av.AudioFrame

    model_config = {"arbitrary_types_allowed": True}
