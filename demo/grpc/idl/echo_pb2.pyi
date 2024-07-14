from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class EchoRequest(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class EchoResponse(_message.Message):
    __slots__ = ("echo",)
    ECHO_FIELD_NUMBER: _ClassVar[int]
    echo: str
    def __init__(self, echo: _Optional[str] = ...) -> None: ...
