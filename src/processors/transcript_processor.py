from typing import Literal
from pydantic import BaseModel

from apipeline.processors.frame_processor import FrameDirection, FrameProcessor

from src.types.frames.data_frames import Frame, InterimTranscriptionFrame, TranscriptionFrame, TransportMessageFrame


class UserTranscriptionMessageData(BaseModel):
    text: str
    user_id: str
    timestamp: str
    final: bool


class UserTranscriptionMessage(BaseModel):
    type: Literal["user-transcription"] = "user-transcription"
    data: UserTranscriptionMessageData


class TranscriptFrameProcessor(FrameProcessor):
    def __init__(self, direction: FrameDirection = FrameDirection.DOWNSTREAM, **kwargs):
        super().__init__(**kwargs)
        self._direction = direction

    async def _push_transport_message(self, model: BaseModel, exclude_none: bool = True):
        frame = TransportMessageFrame(message=model.model_dump(exclude_none=exclude_none))
        await self.push_frame(frame, self._direction)


class UserTranscriptionProcessor(TranscriptFrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        await self.push_frame(frame, direction)

        if isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            await self._handle_user_transcriptions(frame)

    async def _handle_user_transcriptions(self, frame: Frame):
        message = None
        if isinstance(frame, TranscriptionFrame):
            message = UserTranscriptionMessage(
                data=UserTranscriptionMessageData(
                    text=frame.text, user_id=frame.user_id, timestamp=frame.timestamp, final=True
                )
            )
        elif isinstance(frame, InterimTranscriptionFrame):
            message = UserTranscriptionMessage(
                data=UserTranscriptionMessageData(
                    text=frame.text, user_id=frame.user_id, timestamp=frame.timestamp, final=False
                )
            )

        if message:
            await self._push_transport_message(message)
