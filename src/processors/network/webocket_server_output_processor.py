import io
import logging
import wave

import websockets
from apipeline.frames.data_frames import AudioRawFrame
import websockets.connection

from src.processors.audio_camera_output_processor import AudioCameraOutputProcessor
from src.types.network.websocket import WebsocketServerParams


class WebsocketServerOutputProcessor(AudioCameraOutputProcessor):
    def __init__(self, params: WebsocketServerParams, **kwargs):
        super().__init__(params, **kwargs)

        self._params = params

        self._websocket: websockets.WebSocketServerProtocol | None = None

        self._websocket_audio_buffer = bytes()

    async def set_client_connection(self, websocket: websockets.WebSocketServerProtocol | None):
        if self._websocket:
            await self._websocket.close()
            logging.warning("Only one client allowed, using new connection")
        self._websocket = websocket

    async def write_raw_audio_frames(self, frames: bytes):
        if not self._websocket:
            return

        self._websocket_audio_buffer += frames
        while len(self._websocket_audio_buffer) >= self._params.audio_frame_size:
            frame = AudioRawFrame(
                audio=self._websocket_audio_buffer[: self._params.audio_frame_size],
                sample_rate=self._params.audio_out_sample_rate,
                num_channels=self._params.audio_out_channels,
            )

            if self._params.add_wav_header:
                content = io.BytesIO()
                ww = wave.open(content, "wb")
                ww.setsampwidth(frame.sample_width)
                ww.setnchannels(frame.num_channels)
                ww.setframerate(frame.sample_rate)
                ww.writeframes(frame.audio)
                ww.close()
                content.seek(0)
                wav_frame = AudioRawFrame(
                    content.read(), sample_rate=frame.sample_rate, num_channels=frame.num_channels
                )
                frame = wav_frame

            proto = self._params.serializer.serialize(frame)
            if proto:
                await self._websocket.send(proto)

            self._websocket_audio_buffer = self._websocket_audio_buffer[
                self._params.audio_frame_size :
            ]
