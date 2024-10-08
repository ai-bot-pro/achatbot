import logging

from apipeline.frames.sys_frames import MetricsFrame, CancelFrame
from apipeline.frames.data_frames import ImageRawFrame
from apipeline.frames.control_frames import StartFrame, EndFrame

from src.services.livekit_client import LivekitTransportClient
from src.processors.audio_camera_output_processor import AudioCameraOutputProcessor
from src.common.types import DailyParams
from src.types.frames.data_frames import TransportMessageFrame, LivekitTransportMessageFrame


class LivekitOutputTransportProcessor(AudioCameraOutputProcessor):

    def __init__(self, client: LivekitTransportClient, params: DailyParams, **kwargs):
        super().__init__(params, **kwargs)
        self._client = client

    async def start(self, frame: StartFrame):
        # Parent start.
        await super().start(frame)
        # Join the room.
        await self._client.join()

    async def stop(self, frame: EndFrame):
        # Parent stop.
        await super().stop(EndFrame)
        # Leave the room.
        await self._client.leave()

    async def cancel(self, frame: CancelFrame):
        # Parent stop.
        await super().cancel(frame)
        # Leave the room.
        await self._client.leave()

    async def cleanup(self):
        await super().cleanup()
        await self._client.cleanup()

    async def send_message(self, frame: TransportMessageFrame):
        await self._client.send_message(frame)

    async def send_metrics(self, frame: MetricsFrame):
        metrics = {}
        if frame.ttfb:
            metrics["ttfb"] = frame.ttfb
        if frame.processing:
            metrics["processing"] = frame.processing
        if frame.tokens:
            metrics["tokens"] = frame.tokens
        if frame.characters:
            metrics["characters"] = frame.characters

        message = LivekitTransportMessageFrame(message={
            "type": "chatbot-metrics",
            "metrics": metrics
        })
        await self._client.send_message(message)

    async def write_raw_audio_frames(self, frames: bytes):
        await self._client.write_raw_audio_frames(frames)

    async def write_frame_to_camera(self, frame: ImageRawFrame):
        await self._client.write_frame_to_camera(frame)
