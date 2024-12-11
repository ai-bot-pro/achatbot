from dataclasses import dataclass
from typing import Any, Optional

from apipeline.frames.control_frames import ControlFrame

from src.common.types import VADAnalyzerArgs
from src.types.speech.language import Language

#
# Control frames
#


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
    context: Optional[Any] = None  # for openai llm context

    def __str__(self):
        return f"{self.name}, user: {self.user_id}"


@dataclass
class BotStartedSpeakingFrame(ControlFrame):
    """Emitted upstream by transport outputs to indicate the bot started speaking."""

    pass


@dataclass
class BotStoppedSpeakingFrame(ControlFrame):
    """Emitted upstream by transport outputs to indicate the bot stopped speaking."""

    pass


@dataclass
class BotSpeakingFrame(ControlFrame):
    """Emitted by transport outputs while the bot is still speaking. This can be
    used, for example, to detect when a user is idle. That is, while the bot is
    speaking we don't want to trigger any user idle timeout since the user might
    be listening.

    """

    pass


@dataclass
class LLMModelUpdateFrame(ControlFrame):
    """A control frame containing a request to update to a new LLM model."""

    model: str


@dataclass
class TTSVoiceUpdateFrame(ControlFrame):
    """A control frame containing a request to update to a new TTS voice."""

    voice: str


@dataclass
class VADParamsUpdateFrame(ControlFrame):
    """A control frame containing a request to update VAD params. Intended
    to be pushed upstream from RTVI processor.
    """

    params: VADAnalyzerArgs


@dataclass
class ASRModelUpdateFrame(ControlFrame):
    """A control frame containing a request to update the ASR model and optional
    language.
    """

    model: str


@dataclass
class ASRLanguageUpdateFrame(ControlFrame):
    """A control frame containing a request to update to ASR language."""

    language: Language


@dataclass
class ASRArgsUpdateFrame(ControlFrame):
    """A control frame containing a request to update to ASR args."""

    args: dict


@dataclass
class TTSLanguageUpdateFrame(ControlFrame):
    """A control frame containing a request to update to a new TTS language and
    optional voice.
    """

    language: Language


@dataclass
class TTSArgsUpdateFrame(ControlFrame):
    """A control frame containing a request to update to TTS args."""

    args: dict
