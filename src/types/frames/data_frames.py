import json
from dataclasses import dataclass, field
from typing import Any, List


from apipeline.frames.data_frames import Frame, DataFrame, TextFrame, ImageRawFrame, AudioRawFrame

from src.common.types import VADState


@dataclass
class InputImageRawFrame(ImageRawFrame):
    """
    input image frame
    """


@dataclass
class OutputImageRawFrame(ImageRawFrame):
    """
    output image frame
    """


@dataclass
class URLImageRawFrame(ImageRawFrame):
    """An image with an associated URL. Will be shown by the transport if the
    transport's camera is enabled.

    """

    url: str | None

    def __str__(self):
        return f"{self.name}(url: {self.url}, size: {self.size}, format: {self.format}), mode:{self.mode}"


@dataclass
class VisionImageRawFrame(ImageRawFrame):
    """An image with an associated text to ask for a description of it. Will be
    shown by the transport if the transport's camera is enabled.

    """

    text: str | None

    def __str__(self):
        return f"{self.name}(text: {self.text}, size: {self.size}, format: {self.format}, bytes_len:{len(self.image)}, mode:{self.mode}"


@dataclass
class UserImageRawFrame(ImageRawFrame):
    """An image associated to a user. Will be shown by the transport if the
    transport's camera is enabled.

    """

    user_id: str

    def __str__(self):
        return f"{self.name}(user: {self.user_id}, size: {self.size}, format: {self.format}), mode:{self.mode}"


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
    language: str | None = None
    speech_id: int = 0  # active speech segment id
    start_at_s: float = 0.0  # start time point from record start
    end_at_s: float = 0.0  # end time point from record start

    def __str__(self):
        return f"{self.name}(user: {self.user_id}, text: {self.text}, timestamp: {self.timestamp}, language: {self.language} speech_id:{self.speech_id} start_at_s:{self.start_at_s} end_at_s:{self.end_at_s})"


@dataclass
class InterimTranscriptionFrame(TextFrame):
    """A text frame with interim transcription-specific data. Will be placed in
    the transport's receive queue when a participant speaks."""

    user_id: str
    timestamp: str
    language: str

    def __str__(self):
        return f"{self.name}(user: {self.user_id}, text: {self.text}, timestamp: {self.timestamp}, language: {self.language})"


@dataclass
class ASRLiveTranscriptionFrame(TextFrame):
    """
    user_id: as session id
    timestamp: transcript time
    language: Language
    timestamps: list of word start_time or tuple(start_time,end_time)
    is_final: active speech segement final
    speech_id: active speech segment id
    is_final: is end stat
    start_at_s: start time point from record start
    cur_at_s: current time point from record start
    end_at_s: end time point from record start
    """

    user_id: str = ""
    timestamp: str = ""
    language: str | None = None
    timestamps: list = field(default_factory=list)
    speech_id: int = 0
    is_final: bool = False
    start_at_s: float = 0.0
    cur_at_s: float = 0.0
    end_at_s: float = 0.0

    def __str__(self):
        return f"{self.name}(user: {self.user_id}, text: {self.text}, timestamp: {self.timestamp}, language: {self.language}, len(timestamps): {len(self.timestamps)} speech_id: {self.speech_id} is_final: {self.is_final} speech_id: {self.speech_id} start_at_s: {self.start_at_s} cur_at_s: {self.cur_at_s} end_at_s: {self.end_at_s})"


@dataclass
class TranslationStreamingFrame(TextFrame):
    is_final: bool = False

    def __str__(self):
        return f"{super().__str__()} is_final: {self.is_final}"


@dataclass
class TranslationFrame(TextFrame):
    src_lang: str = ""
    target_lang: str = ""
    src_text: str = ""

    def __str__(self):
        return f"{self.name}(src_lang: {self.src_lang}, target_lang: {self.target_lang}, src_text: {self.src_text}, target_text: {self.text})"


@dataclass
class LLMMessagesFrame(DataFrame):
    """A frame containing a list of LLM messages. Used to signal that an LLM
    service should run a chat completion and emit an LLMStartFrames, TextFrames
    and an LLMEndFrame. Note that the messages property on this class is
    mutable, and will be be updated by various ResponseAggregator frame
    processors.

    """

    messages: List[dict]

    def __str__(self):
        return f"{self.name}(messages: {self.messages})"


@dataclass
class TransportMessageFrame(DataFrame):
    message: Any
    urgent: bool = False

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
class LLMSetToolsFrame(DataFrame):
    """A frame containing a list of tools for an LLM to use for function calling.
    The specific format depends on the LLM being used, but it should typically
    contain JSON Schema objects.
    """

    tools: List[dict]


@dataclass
class TTSSpeakFrame(DataFrame):
    """A frame that contains a text that should be spoken by the TTS in the
    pipeline (if any).

    """

    text: str


@dataclass
class DailyTransportMessageFrame(TransportMessageFrame):
    participant_id: str | None = None


@dataclass
class LivekitTransportMessageFrame(TransportMessageFrame):
    participant_id: str | None = None


@dataclass
class AgoraTransportMessageFrame(TransportMessageFrame):
    participant_id: str | None = None


@dataclass
class FunctionCallResultFrame(DataFrame):
    """A frame containing the result of an LLM function (tool) call."""

    function_name: str
    tool_call_id: str
    arguments: dict
    result: Any


@dataclass
class InputAudioRawFrame(AudioRawFrame):
    """Input audio frame"""


@dataclass
class OutputAudioRawFrame(AudioRawFrame):
    """output audio frame"""


@dataclass
class VADStateAudioRawFrame(AudioRawFrame):
    """VAD state audio frame"""

    state: VADState = VADState.QUIET  # Starting/(start)/Speaking/Stopping/(end)/Quiet
    speech_id: int = 0  # active speech segment id
    is_final: bool = False  # is end stat
    start_at_s: float = 0.0  # start time point from record start
    cur_at_s: float = 0.0  # current time point from record start
    end_at_s: float = 0.0  # end time point from record start

    def __str__(self):
        return f"{super().__str__()} (state: {self.state} speech_id: {self.speech_id} is_final: {self.is_final} speech_id: {self.speech_id} start_at_s: {self.start_at_s} cur_at_s: {self.cur_at_s} end_at_s: {self.end_at_s})"


@dataclass
class PathAudioRawFrame(AudioRawFrame):
    """An audio with saved path."""

    path: str = ""

    def __post_init__(self):
        super().__post_init__()

    def __str__(self):
        return f"path:{self.path} {super().__str__()}"


@dataclass
class VADAudioRawFrame(AudioRawFrame):
    speech_id: int = 0  # active speech segment id
    start_at_s: float = 0.0  # start time point from record start
    end_at_s: float = 0.0  # end time point from record start

    def __str__(self):
        return f"speech_id:{self.speech_id} start_at_s:{self.start_at_s} end_at_s:{self.end_at_s} {super().__str__()}"


@dataclass
class UserAudioRawFrame(AudioRawFrame):
    """An audio associated to a user. Will be shown by the transport if the
    transport's audio is enabled.

    """

    user_id: str = ""

    def __post_init__(self):
        super().__post_init__()

    def __str__(self):
        return f"user_id:{self.user_id} {super().__str__()}"


@dataclass
class UserVoiceRawFrame(UserAudioRawFrame):
    """
    An user audio associated to llm response text.
    """

    text: str = ""

    def __post_init__(self):
        super().__post_init__()

    def __str__(self):
        return f"text:{self.text} {super().__str__()}"


@dataclass
class VisionImageVoiceRawFrame(DataFrame):
    """An image + audio with an instruct text to ask for a description of it. Will be
    shown by the transport if the transport's camera is enabled.

    """

    text: str | None = None
    audio: AudioRawFrame | None = None
    images: List[ImageRawFrame] = field(default_factory=list)

    def __str__(self):
        s = f"{self.name}(text: {self.text}, audio:{self.audio}, images:"
        for image in self.images:
            s += f"{image}, "
        s += ")"
        return s


@dataclass
class UserVisionImageVoiceRawFrame(VisionImageVoiceRawFrame):
    """An user image + audio with an instruct text to ask for a description of it. Will be
    shown by the transport if the transport's camera is enabled.

    """

    user_id: str = ""

    def __str__(self):
        return f"user_id:{self.user_id} {super().__str__()}"


@dataclass
class AnimationAudioRawFrame(AudioRawFrame):
    animation_json: str = "{}"
    avatar_status: str = ""

    def __str__(self):
        super_str = super().__str__()
        return (
            f"{super_str} animation_json: {self.animation_json} avatar_status: {self.avatar_status}"
        )


@dataclass
class TextQuestionsAudioRawFrame(AudioRawFrame, TextFrame):
    """text questions with audio frame"""


@dataclass
class FunctionCallFrame(Frame):
    """llm gened function call frame"""

    tool_call_id: str = ""
    function_name: str = ""
    arguments: str = ""
    index: int = 0
    type: str = "function"

    def __str__(self):
        return f"{self.name}(function_name: {self.function_name}, tool_call_id: {self.tool_call_id}, arguments: {self.arguments} index: {self.index} type: {self.type})"

    @property
    def arguments_dict(self):
        return json.loads(self.arguments)


@dataclass
class ReasoningThinkTextFrame(Frame):
    """llm gen completed reasoning think text tokens frame"""

    text: str = ""

    def __str__(self):
        return f"{self.name}(text: {self.text})"


@dataclass
class LLMGenedTokensFrame(Frame):
    """llm gened tokens frame"""

    token_ids: list[int] = field(default_factory=list)
    text_tokens: list[str] = field(default_factory=list)
    audio_tokens: list[str] = field(default_factory=list)
    tool_calls: list[FunctionCallFrame] = field(default_factory=list)

    def __str__(self):
        return f"{self.name}(token_ids: {self.token_ids} text_tokens: {self.text_tokens} audio_tokens: {self.audio_tokens} tool_calls: {self.tool_calls})"
