from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class LoadModelRequest(_message.Message):
    __slots__ = ("tts_tag", "is_reload", "json_kwargs")
    TTS_TAG_FIELD_NUMBER: _ClassVar[int]
    IS_RELOAD_FIELD_NUMBER: _ClassVar[int]
    JSON_KWARGS_FIELD_NUMBER: _ClassVar[int]
    tts_tag: str
    is_reload: bool
    json_kwargs: str
    def __init__(self, tts_tag: _Optional[str] = ..., is_reload: bool = ..., json_kwargs: _Optional[str] = ...) -> None: ...

class LoadModelResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SynthesizeRequest(_message.Message):
    __slots__ = ("tts_text", "json_kwargs")
    TTS_TEXT_FIELD_NUMBER: _ClassVar[int]
    JSON_KWARGS_FIELD_NUMBER: _ClassVar[int]
    tts_text: str
    json_kwargs: str
    def __init__(self, tts_text: _Optional[str] = ..., json_kwargs: _Optional[str] = ...) -> None: ...

class SynthesizeResponse(_message.Message):
    __slots__ = ("tts_audio",)
    TTS_AUDIO_FIELD_NUMBER: _ClassVar[int]
    tts_audio: bytes
    def __init__(self, tts_audio: _Optional[bytes] = ...) -> None: ...

class GetVoicesRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetVoicesResponse(_message.Message):
    __slots__ = ("voices",)
    VOICES_FIELD_NUMBER: _ClassVar[int]
    voices: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, voices: _Optional[_Iterable[str]] = ...) -> None: ...

class SetVoiceRequest(_message.Message):
    __slots__ = ("voice",)
    VOICE_FIELD_NUMBER: _ClassVar[int]
    voice: str
    def __init__(self, voice: _Optional[str] = ...) -> None: ...

class SetVoiceResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetStreamInfoRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetStreamInfoReponse(_message.Message):
    __slots__ = ("format", "channels", "rate", "sample_width")
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    CHANNELS_FIELD_NUMBER: _ClassVar[int]
    RATE_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_WIDTH_FIELD_NUMBER: _ClassVar[int]
    format: int
    channels: int
    rate: int
    sample_width: int
    def __init__(self, format: _Optional[int] = ..., channels: _Optional[int] = ..., rate: _Optional[int] = ..., sample_width: _Optional[int] = ...) -> None: ...
