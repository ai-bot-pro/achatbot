from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel


class AvatarStatus(Enum):
    SPEAKING = 0
    LISTENING = 1


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

    def get_algo_audio_duration(self) -> float:
        return (
            len(self.algo_audio_data) / self.algo_audio_sample_rate / 2
            if self.algo_audio_data
            else 0
        )

    def __str__(self):
        return f"{self.speech_id=} {self.get_audio_duration()=} {self.get_algo_audio_duration()=} {self.end_of_speech=} {self.front_padding_duration=} {self.end_padding_duration=}"


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
