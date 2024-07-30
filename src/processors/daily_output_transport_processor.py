import logging

from apipeline.frames.sys_frames import MetricsFrame
from apipeline.frames.data_frames import ImageRawFrame
from apipeline.frames.control_frames import StartFrame

from src.services.daily.client import DailyTransportClient
from src.processors.audio_camera_output_processor import AudioCameraOutputProcessor
from src.common.types import DailyParams, DailyTransportMessageFrame


class DailyOutputTransportProcessor(AudioCameraOutputProcessor):

    def __init__(self, client: DailyTransportClient, params: DailyParams, **kwargs):
        super().__init__(params, **kwargs)

        self._client = client

    async def start(self, frame: StartFrame):
        # Parent start.
        await super().start(frame)
        # Join the room.
        await self._client.join()

    async def stop(self):
        # Parent stop.
        await super().stop()
        # Leave the room.
        await self._client.leave()

    async def cleanup(self):
        await super().cleanup()
        await self._client.cleanup()

    async def send_message(self, frame: DailyTransportMessageFrame):
        await self._client.send_message(frame)

    async def send_metrics(self, frame: MetricsFrame):
        ttfb = [{"name": n, "time": t} for n, t in frame.ttfb.items()]
        message = DailyTransportMessageFrame(message={
            "type": "pipecat-metrics",
            "metrics": {
                "ttfb": ttfb
            },
        })
        await self._client.send_message(message)

    async def write_raw_audio_frames(self, frames: bytes):
        await self._client.write_raw_audio_frames(frames)

    async def write_frame_to_camera(self, frame: ImageRawFrame):
        await self._client.write_frame_to_camera(frame)
