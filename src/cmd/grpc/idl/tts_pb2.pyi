from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class LoadModelRequest(_message.Message):
    __slots__ = ("tts_tag", "is_reload", "kwargs")
    class KwargsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    TTS_TAG_FIELD_NUMBER: _ClassVar[int]
    IS_RELOAD_FIELD_NUMBER: _ClassVar[int]
    KWARGS_FIELD_NUMBER: _ClassVar[int]
    tts_tag: str
    is_reload: bool
    kwargs: _containers.ScalarMap[str, str]
    def __init__(self, tts_tag: _Optional[str] = ..., is_reload: bool = ..., kwargs: _Optional[_Mapping[str, str]] = ...) -> None: ...

class LoadModelResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SynthesizeRequest(_message.Message):
    __slots__ = ("tts_text",)
    TTS_TEXT_FIELD_NUMBER: _ClassVar[int]
    tts_text: str
    def __init__(self, tts_text: _Optional[str] = ...) -> None: ...

class SynthesizeResponse(_message.Message):
    __slots__ = ("tts_audio",)
    TTS_AUDIO_FIELD_NUMBER: _ClassVar[int]
    tts_audio: bytes
    def __init__(self, tts_audio: _Optional[bytes] = ...) -> None: ...
