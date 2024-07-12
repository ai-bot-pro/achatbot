from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class SynthesizeRequest(_message.Message):
    __slots__ = ("tts_text", "tts_tag", "kawa_params")
    class KawaParamsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    TTS_TEXT_FIELD_NUMBER: _ClassVar[int]
    TTS_TAG_FIELD_NUMBER: _ClassVar[int]
    KAWA_PARAMS_FIELD_NUMBER: _ClassVar[int]
    tts_text: str
    tts_tag: str
    kawa_params: _containers.ScalarMap[str, str]
    def __init__(self, tts_text: _Optional[str] = ..., tts_tag: _Optional[str] = ..., kawa_params: _Optional[_Mapping[str, str]] = ...) -> None: ...

class SynthesizeResponse(_message.Message):
    __slots__ = ("tts_audio",)
    TTS_AUDIO_FIELD_NUMBER: _ClassVar[int]
    tts_audio: bytes
    def __init__(self, tts_audio: _Optional[bytes] = ...) -> None: ...
