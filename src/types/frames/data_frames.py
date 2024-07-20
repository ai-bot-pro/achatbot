from typing import Any, List, Mapping, Tuple
from dataclasses import dataclass, field


from .base import Frame


@dataclass
class DataFrame(Frame):
    pass


@dataclass
class AudioRawFrame(DataFrame):
    """A chunk of audio. Will be played by the transport if the transport's
    microphone has been enabled.

    """
    audio: bytes
    sample_rate: int
    num_channels: int

    def __post_init__(self):
        super().__post_init__()
        self.num_frames = int(len(self.audio) / (self.num_channels * 2))

    def __str__(self):
        return f"{self.name}(size: {len(self.audio)}, frames: {self.num_frames}, sample_rate: {self.sample_rate}, channels: {self.num_channels})"


@dataclass
class ImageRawFrame(DataFrame):
    """An image. Will be shown by the transport if the transport's camera is
    enabled.

    """
    image: bytes
    size: Tuple[int, int]
    format: str | None

    def __str__(self):
        return f"{self.name}(size: {self.size}, format: {self.format})"


@dataclass
class URLImageRawFrame(ImageRawFrame):
    """An image with an associated URL. Will be shown by the transport if the
    transport's camera is enabled.

    """
    url: str | None

    def __str__(self):
        return f"{self.name}(url: {self.url}, size: {self.size}, format: {self.format})"


@dataclass
class VisionImageRawFrame(ImageRawFrame):
    """An image with an associated text to ask for a description of it. Will be
    shown by the transport if the transport's camera is enabled.

    """
    text: str | None

    def __str__(self):
        return f"{self.name}(text: {self.text}, size: {self.size}, format: {self.format})"


@dataclass
class UserImageRawFrame(ImageRawFrame):
    """An image associated to a user. Will be shown by the transport if the
    transport's camera is enabled.

    """
    user_id: str

    def __str__(self):
        return f"{self.name}(user: {self.user_id}, size: {self.size}, format: {self.format})"


@dataclass
class SpriteFrame(Frame):
    """An animated sprite. Will be shown by the transport if the transport's
    camera is enabled. Will play at the framerate specified in the transport's
    `fps` constructor parameter.

    """
    images: List[ImageRawFrame]

    def __str__(self):
        return f"{self.name}(size: {len(self.images)})"


@dataclass
class TextFrame(DataFrame):
    """A chunk of text. Emitted by LLM services, consumed by TTS services, can
    be used to send text through pipelines.

    """
    text: str

    def __str__(self):
        return f"{self.name}(text: {self.text})"


@dataclass
class TranscriptionFrame(TextFrame):
    """A text frame with transcription-specific data. Will be placed in the
    transport's receive queue when a participant speaks.

    """
    user_id: str
    timestamp: str

    def __str__(self):
        return f"{self.name}(user: {self.user_id}, text: {self.text}, timestamp: {self.timestamp})"


@dataclass
class InterimTranscriptionFrame(TextFrame):
    """A text frame with interim transcription-specific data. Will be placed in
    the transport's receive queue when a participant speaks."""
    user_id: str
    timestamp: str

    def __str__(self):
        return f"{self.name}(user: {self.user_id}, text: {self.text}, timestamp: {self.timestamp})"


@dataclass
class LLMMessagesFrame(DataFrame):
    """A frame containing a list of LLM messages. Used to signal that an LLM
    service should run a chat completion and emit an LLMStartFrames, TextFrames
    and an LLMEndFrame. Note that the messages property on this class is
    mutable, and will be be updated by various ResponseAggregator frame
    processors.

    """
    messages: List[dict]


@dataclass
class TransportMessageFrame(DataFrame):
    message: Any

    def __str__(self):
        return f"{self.name}(message: {self.message})"
