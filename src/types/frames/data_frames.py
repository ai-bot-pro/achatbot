from dataclasses import dataclass
from typing import Any, List


from apipeline.frames.data_frames import Frame, DataFrame, TextFrame, ImageRawFrame


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


@dataclass
class LLMMessagesAppendFrame(DataFrame):
    """A frame containing a list of LLM messages that neeed to be added to the
    current context.

    """
    messages: List[dict]


@dataclass
class LLMMessagesUpdateFrame(DataFrame):
    """A frame containing a list of new LLM messages. These messages will
    replace the current context LLM messages and should generate a new
    LLMMessagesFrame.

    """
    messages: List[dict]


@dataclass
class TTSSpeakFrame(DataFrame):
    """A frame that contains a text that should be spoken by the TTS in the
    pipeline (if any).

    """
    text: str


@dataclass
class DailyTransportMessageFrame(TransportMessageFrame):
    participant_id: str | None = None
