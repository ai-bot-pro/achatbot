import logging

from apipeline.frames import StartFrame, EndFrame, CancelFrame

from src.types.frames.data_frames import (
    ImageRawFrame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    TransportMessageFrame,
)
from src.common.types import AudioCameraParams
from src.services.small_webrtc_client import SmallWebRTCClient
from src.processors.audio_camera_output_processor import AudioCameraOutputProcessor


class SmallWebRTCOutputProcessor(AudioCameraOutputProcessor):
    def __init__(
        self,
        client: SmallWebRTCClient,
        params: AudioCameraParams,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._client = client
        self._params = params

        # Whether we have seen a StartFrame already.
        self._initialized = False

    async def start(self, frame: StartFrame):
        if self._initialized:
            return

        self._initialized = True

        await self._client.setup(self._params, frame)
        await self._client.connect()
        await super().start(frame)

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._client.disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._client.disconnect()

    async def send_message(self, frame: TransportMessageFrame):
        await self._client.send_message(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame):
        await self._client.write_audio_frame(frame)

    async def write_raw_audio_frames(self, chunk: bytes):
        if len(chunk) < self._audio_chunk_size:
            pad_bytes = b"\x00" * (self._audio_chunk_size - len(chunk))
            chunk += pad_bytes
        await self._client.write_raw_audio_frames(chunk)

    async def write_video_frame(self, frame: OutputImageRawFrame):
        await self._client.write_video_frame(frame)

    async def write_frame_to_camera(self, frame: ImageRawFrame):
        await self._client.write_video_frame(frame)
