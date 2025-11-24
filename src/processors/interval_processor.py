import asyncio
import logging
from typing import List

from apipeline.frames.control_frames import StartFrame, EndFrame
from apipeline.frames.sys_frames import Frame, CancelFrame
from apipeline.processors.frame_processor import FrameDirection
from apipeline.processors.async_frame_processor import AsyncFrameProcessor

from src.types.frames.control_frames import IntervalFrame


class IntervalProcessor(AsyncFrameProcessor):
    """window.setInterval for IntervalFrame"""

    def __init__(
        self,
        *,
        interval_time_ms: int,
        types: List[type] = [],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.interval_time_ms = interval_time_ms
        self.types = types
        self._interval_task = None

    async def start(self):
        self._interval_task = self.get_event_loop().create_task(self._interval_handle())

    async def end(self):
        self._interval_task.cancel()
        await self._interval_task

    async def cancel(self):
        self._interval_task.cancel()
        await self._interval_task

    async def _interval_handle(self):
        while True:
            try:
                await asyncio.sleep(self.interval_time_ms / 1000)
                await self.push_frame(
                    IntervalFrame(
                        interval_time_ms=self.interval_time_ms,
                        types=self.types,
                    )
                )
            except Exception as e:
                logging.error(f"IntervalProcessor error: {e}")
            except asyncio.CancelledError:
                logging.info("IntervalProcessor cancelled")
                break

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            await self.start()
        elif isinstance(frame, EndFrame):
            await self.push_frame(frame, direction)
            await self.end()
        elif isinstance(frame, CancelFrame):
            await self.cancel()
            await self.push_frame(frame, direction)
        else:
            await self.queue_frame(frame, direction)
