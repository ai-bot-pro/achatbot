from typing import Awaitable, Callable

from pydantic import BaseModel
from fastapi import WebSocket
from apipeline.serializers.protobuf import ProtobufFrameSerializer, FrameSerializer

from src.common.types import AudioCameraParams


class FastapiWebsocketServerParams(AudioCameraParams):
    add_wav_header: bool = False
    audio_out_frame_ms: int = 200  # 200ms
    serializer: FrameSerializer = ProtobufFrameSerializer()


class FastapiWebsocketServerCallbacks(BaseModel):
    on_client_connected: Callable[[WebSocket], Awaitable[None]]
    on_client_disconnected: Callable[[WebSocket], Awaitable[None]]
