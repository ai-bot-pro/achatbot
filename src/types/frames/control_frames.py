from dataclasses import dataclass

from .base import Frame

#
# Control frames
#


@dataclass
class ControlFrame(Frame):
    pass


@dataclass
class EndFrame(ControlFrame):
    """Indicates that a pipeline has ended and frame processors and pipelines
    should be shut down. If the transport receives this frame, it will stop
    sending frames to its output channel(s) and close all its threads. Note,
    that this is a control frame, which means it will received in the order it
    was sent (unline system frames).

    """
    pass


@dataclass
class LLMFullResponseStartFrame(ControlFrame):
    """Used to indicate the beginning of a full LLM response. Following
    LLMResponseStartFrame, TextFrame and LLMResponseEndFrame for each sentence
    until a LLMFullResponseEndFrame."""
    pass


@dataclass
class LLMFullResponseEndFrame(ControlFrame):
    """Indicates the end of a full LLM response."""
    pass


@dataclass
class LLMResponseStartFrame(ControlFrame):
    """Used to indicate the beginning of an LLM response. Following TextFrames
    are part of the LLM response until an LLMResponseEndFrame"""
    pass


@dataclass
class LLMResponseEndFrame(ControlFrame):
    """Indicates the end of an LLM response."""
    pass


@dataclass
class UserStartedSpeakingFrame(ControlFrame):
    """Emitted by VAD to indicate that a user has started speaking. This can be
    used for interruptions or other times when detecting that someone is
    speaking is more important than knowing what they're saying (as you will
    with a TranscriptionFrame)

    """
    pass


@dataclass
class UserStoppedSpeakingFrame(ControlFrame):
    """Emitted by the VAD to indicate that a user stopped speaking."""
    pass


@dataclass
class TTSStartedFrame(ControlFrame):
    """Used to indicate the beginning of a TTS response. Following
    AudioRawFrames are part of the TTS response until an TTSEndFrame. These
    frames can be used for aggregating audio frames in a transport to optimize
    the size of frames sent to the session, without needing to control this in
    the TTS service.

    """
    pass


@dataclass
class TTSStoppedFrame(ControlFrame):
    """Indicates the end of a TTS response."""
    pass


@dataclass
class UserImageRequestFrame(ControlFrame):
    """A frame user to request an image from the given user."""
    user_id: str

    def __str__(self):
        return f"{self.name}, user: {self.user_id}"
