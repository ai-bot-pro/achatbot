from typing import Awaitable, Callable
from pydantic import BaseModel

import websockets
from apipeline.serializers.protobuf import ProtobufFrameSerializer, FrameSerializer

from src.common.types import AudioCameraParams


class WebsocketServerParams(AudioCameraParams):
    add_wav_header: bool = False
    audio_frame_size: int = 6400  # 200ms with 16K hz 1 channel 2 sample_width
    serializer: FrameSerializer = ProtobufFrameSerializer()


class WebsocketServerCallbacks(BaseModel):
    on_client_connected: Callable[[websockets.WebSocketServerProtocol], Awaitable[None]]
    on_client_disconnected: Callable[[websockets.WebSocketServerProtocol], Awaitable[None]]
