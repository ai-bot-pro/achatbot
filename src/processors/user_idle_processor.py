from typing import Awaitable, Callable

from apipeline.frames.sys_frames import Frame
from apipeline.processors.user_idle_processor import UserIdleProcessor
from apipeline.processors.frame_processor import FrameDirection

from src.types.frames.control_frames import (
    BotSpeakingFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)


class UserIdleProcessor(UserIdleProcessor):
    """This class is based UserIdleProcessor,
    for BotSpeakingFrame,UserStartedSpeakingFrame,UserStoppedSpeakingFrame

    timeout to call the callback.
    """

    def __init__(
        self,
        *,
        callback: Callable[["UserIdleProcessor"], Awaitable[None]],
        timeout: float,
        **kwargs,
    ):
        super().__init__(callback=callback, timeout=timeout, **kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # We shouldn't call the idle callback if the user or the bot are speaking.
        if isinstance(frame, UserStartedSpeakingFrame):
            self._interrupted = True
            self._idle_event.set()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._interrupted = False
            self._idle_event.set()
        elif isinstance(frame, BotSpeakingFrame):
            self._idle_event.set()
