from typing import Literal, Optional
from pydantic import BaseModel

from apipeline.processors.frame_processor import FrameDirection, FrameProcessor

from src.types.frames.data_frames import (
    Frame,
    TextFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TransportMessageFrame,
)
from src.types.frames.control_frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)


class TextMessageData(BaseModel):
    text: str
    final: Optional[bool] = None


class BotTranscriptionMessage(BaseModel):
    type: Literal["bot-llm-text", "bot-tts-text"] = "bot-llm-text"
    data: TextMessageData


class UserTranscriptionMessageData(BaseModel):
    text: str
    user_id: str
    timestamp: str
    final: bool


class UserTranscriptionMessage(BaseModel):
    type: Literal["user-transcription"] = "user-transcription"
    data: UserTranscriptionMessageData


class AppMessageProcessor(FrameProcessor):
    def __init__(self, direction: FrameDirection = FrameDirection.DOWNSTREAM, **kwargs):
        super().__init__(**kwargs)
        self._direction = direction

    async def _push_transport_message(self, model: BaseModel, exclude_none: bool = True):
        frame = TransportMessageFrame(message=model.model_dump(exclude_none=exclude_none))
        await self.push_frame(frame, self._direction)


class UserTranscriptionProcessor(AppMessageProcessor):
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


class BotLLMTextProcessor(AppMessageProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        await self.push_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            await self._handle_text("", False)
        if isinstance(frame, LLMFullResponseEndFrame):
            await self._handle_text("", True)
        if isinstance(frame, TextFrame):
            await self._handle_text(frame.text, False)

    async def _handle_text(self, text: str, final: bool = False):
        message = BotTranscriptionMessage(
            type="bot-llm-text",
            data=TextMessageData(text=text, final=final),
        )
        await self._push_transport_message(message)


class BotTTSTextProcessor(AppMessageProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        await self.push_frame(frame, direction)

        if isinstance(frame, TextFrame):
            await self._handle_text(frame)

    async def _handle_text(self, frame: TextFrame):
        message = BotTranscriptionMessage(
            type="bot-tts-text",
            data=TextMessageData(text=frame.text),
        )
        await self._push_transport_message(message)


# ------------controller message types---------------------


class UserStartedSpeakingMessage(BaseModel):
    type: Literal["user-started-speaking"] = "user-started-speaking"


class UserStoppedSpeakingMessage(BaseModel):
    type: Literal["user-stopped-speaking"] = "user-stopped-speaking"


class TTSStartedMessage(BaseModel):
    type: Literal["tts-started"] = "tts-started"


class TTSStoppedMessage(BaseModel):
    type: Literal["tts-stopped"] = "tts-stopped"


class BotStartedSpeakingMessage(BaseModel):
    type: Literal["bot-started-speaking"] = "bot-started-speaking"


class BotStoppedSpeakingMessage(BaseModel):
    type: Literal["bot-stopped-speaking"] = "bot-stopped-speaking"


class AppMessageControllProcessor(AppMessageProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        await self.push_frame(frame, direction)

        if isinstance(frame, (UserStartedSpeakingFrame, UserStoppedSpeakingFrame)):
            await self._handle_interruptions(frame)
        elif isinstance(frame, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame)):
            await self._handle_bot_speaking(frame)
        elif isinstance(frame, (TTSStartedFrame, TTSStoppedFrame)):
            await self._handle_tts(frame)

    async def _handle_interruptions(self, frame: Frame):
        message = None
        if isinstance(frame, UserStartedSpeakingFrame):
            message = UserStartedSpeakingMessage()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            message = UserStoppedSpeakingMessage()

        if message:
            await self._push_transport_message(message)

    async def _handle_bot_speaking(self, frame: Frame):
        message = None
        if isinstance(frame, BotStartedSpeakingFrame):
            message = BotStartedSpeakingMessage()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            message = BotStoppedSpeakingMessage()

        if message:
            await self._push_transport_message(message)

    async def _handle_tts(self, frame: Frame):
        message = None
        if isinstance(frame, TTSStartedFrame):
            message = TTSStartedMessage()
        elif isinstance(frame, TTSStoppedFrame):
            message = TTSStoppedMessage()

        if message:
            await self._push_transport_message(message)
