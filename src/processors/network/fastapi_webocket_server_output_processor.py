import io
import wave
import logging

from fastapi import WebSocket
from fastapi.websockets import WebSocketState
from apipeline.frames.data_frames import Frame, AudioRawFrame
from apipeline.frames.sys_frames import StartInterruptionFrame
from apipeline.processors.frame_processor import FrameDirection

from src.processors.audio_camera_output_processor import AudioCameraOutputProcessor
from src.types.network.fastapi_websocket import FastapiWebsocketServerParams


class FastapiWebsocketServerOutputProcessor(AudioCameraOutputProcessor):
    def __init__(self, websocket: WebSocket, params: FastapiWebsocketServerParams, **kwargs):
        super().__init__(params, **kwargs)

        self._websocket = websocket
        self._params = params
        self._websocket_audio_buffer = bytes()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartInterruptionFrame):
            await self._write_frame(frame)

    async def write_raw_audio_frames(self, frames: bytes):
        self._websocket_audio_buffer += frames
        while len(self._websocket_audio_buffer):
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

            payload = self._params.serializer.serialize(frame)
            if payload and self._websocket.client_state == WebSocketState.CONNECTED:
                await self._websocket.send_text(payload)

            self._websocket_audio_buffer = self._websocket_audio_buffer[
                self._params.audio_frame_size:
            ]

    async def _write_frame(self, frame: Frame):
        payload = self._params.serializer.serialize(frame)
        if payload and self._websocket.client_state == WebSocketState.CONNECTED:
            await self._websocket.send_text(payload)