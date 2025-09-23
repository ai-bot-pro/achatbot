import time
import logging

from apipeline.frames import Frame, AudioRawFrame
from apipeline.processors.frame_processor import FrameDirection
from typing import Type

from src.common.interface import ISpeechEnhancer
from src.processors.session_processor import SessionProcessor
from src.types.frames import (
    VADStateAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from src.common.session import Session


class AudioNoiseFilter(SessionProcessor):
    def __init__(
        self,
        se: ISpeechEnhancer,
        start_frame_type: Type[Frame] = UserStartedSpeakingFrame,
        end_frame_type: Type[Frame] = UserStoppedSpeakingFrame,
        session: Session | None = None,
        **kwargs,
    ):
        super().__init__(session=session, **kwargs)
        self._start_frame_type = start_frame_type
        self._end_frame_type = end_frame_type
        self._se = se
        self._user_speaking = False
        self._verbose = kwargs.get("verbose", False)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, self._start_frame_type):
            self._user_speaking = True
        if isinstance(frame, self._end_frame_type):
            self._user_speaking = False

        if isinstance(frame, AudioRawFrame):
            if self._user_speaking is False:
                await self.push_frame(frame)
                return

            self.session.ctx.state["audio_chunk"] = frame.audio
            self.session.ctx.state["sample_rate"] = frame.sample_rate
            self.session.ctx.state["is_last"] = False
            if isinstance(frame, VADStateAudioRawFrame):
                self.session.ctx.state["is_last"] = frame.is_final
            start = time.time()
            filter_audio = self._se.enhance(self.session)  # only read, don't CoW
            frame.audio = filter_audio
            if self._verbose:
                print(
                    f"{self.name} filter audio_chunk_len: {len(self.session.ctx.state['audio_chunk'])}"
                    f" {str(frame)} cost: {time.time() - start} s"
                )
        await self.push_frame(frame)
