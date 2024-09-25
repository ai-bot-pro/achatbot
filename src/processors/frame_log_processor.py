import logging
from typing import Optional

from apipeline.frames.data_frames import Frame, AudioRawFrame
from apipeline.processors.frame_processor import FrameDirection, FrameProcessor

from src.types.frames.control_frames import BotSpeakingFrame
from src.types.frames.data_frames import TransportMessageFrame


class FrameLogger(FrameProcessor):
    def __init__(
            self,
            prefix="Frame",
            color: Optional[str] = None,
            ignored_frame_types: Optional[list] = [
                BotSpeakingFrame,
                AudioRawFrame,
                TransportMessageFrame],
            include_frame_types: Optional[list] = None):
        super().__init__()
        self._prefix = prefix
        self._color = color
        self._ignored_frame_types = tuple(ignored_frame_types) if ignored_frame_types else None
        self._include_frame_types = tuple(include_frame_types) if include_frame_types else None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if self._ignored_frame_types and not isinstance(frame, self._ignored_frame_types):
            if not self._include_frame_types \
                or (self._include_frame_types
                    and isinstance(frame, self._include_frame_types)):
                from_to = f"{self._prev} ---> {self}"
                if direction == FrameDirection.UPSTREAM:
                    from_to = f"{self} <--- {self._next} "
                msg = f"{from_to} {self._prefix}: {frame}"
                if self._color:
                    msg = f"<{self._color}>{msg}</>"
                logging.info(msg)

        await self.push_frame(frame, direction)
