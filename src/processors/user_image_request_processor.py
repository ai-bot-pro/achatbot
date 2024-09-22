
import re
from apipeline.pipeline.pipeline import FrameProcessor
from apipeline.pipeline.task import FrameDirection
from apipeline.frames.data_frames import Frame, TextFrame

from src.types.frames.control_frames import UserImageRequestFrame


class UserImageRequestProcessor(FrameProcessor):
    def __init__(
            self,
            participant_id: str | None = None,
            init_user_prompt: str | list = "show me the money :)",
    ):
        super().__init__()
        self._participant_id = participant_id
        self._init_user_prompt = init_user_prompt

    def set_participant_id(self, participant_id: str):
        self._participant_id = participant_id

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if self._participant_id and isinstance(frame, TextFrame):
            await self.push_frame(UserImageRequestFrame(self._participant_id), FrameDirection.UPSTREAM)
        await self.push_frame(frame, direction)
