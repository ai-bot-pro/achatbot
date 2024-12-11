from typing import Literal

from pydantic import BaseModel
from apipeline.processors.frame_processor import FrameProcessor, FrameDirection

from src.types.frames.data_frames import (
    Frame,
    TextFrame,
    TransportMessageFrame,
)


class RTVITTSTextMessageData(BaseModel):
    text: str


class RTVITTSTextMessage(BaseModel):
    label: Literal["rtvi"] = "rtvi-ai"
    type: Literal["tts-text"] = "tts-text"
    data: RTVITTSTextMessageData


class RTVITTSTextProcessor(FrameProcessor):
    def __init__(self):
        super().__init__()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            message = RTVITTSTextMessage(data=RTVITTSTextMessageData(text=frame.text))
            await self.push_frame(
                TransportMessageFrame(message=message.model_dump(exclude_none=True))
            )

        await self.push_frame(frame, direction)
