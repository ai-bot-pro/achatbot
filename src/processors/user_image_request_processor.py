import time
from typing import Type
from apipeline.pipeline.pipeline import FrameProcessor
from apipeline.pipeline.task import FrameDirection
from apipeline.frames.data_frames import Frame, TextFrame

from src.types.frames.control_frames import UserImageRequestFrame
from src.types.frames.data_frames import UserImageRawFrame, VADStateAudioRawFrame
from src.common.types import VADState


class UserImageBaseProcessor(FrameProcessor):
    def __init__(
        self,
        participant_id: str | None = None,
        init_user_prompts: str | list = "show me the money :)",
        desc_img_prompt: str = "Describe the image in a short sentence.",
        request_frame_cls: Type[Frame] = TextFrame,
    ):
        super().__init__()
        self._participant_id = participant_id
        self._init_user_prompts = init_user_prompts
        self._desc_img_prompt = desc_img_prompt
        self._request_frame_cls = request_frame_cls

    def set_participant_id(self, participant_id: str):
        self._participant_id = participant_id


class UserVADImageRequestProcessor(UserImageBaseProcessor):
    """
    process:
    - VADStateAudioRawFrame for push user image request frame to upstream
    - all frames to downstream
    """

    def __init__(
        self, states: list[VADState] = [VADState.SPEAKING], interval_time_ms: int | None = None
    ):
        super().__init__(request_frame_cls=VADStateAudioRawFrame)
        self._states = states
        self._interval_time_ms = interval_time_ms or 200
        self._push_time = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if (
            self._participant_id
            and isinstance(frame, VADStateAudioRawFrame)
            and frame.state in self._states
        ):
            if time.time() - self._push_time >= self._interval_time_ms / 1000:
                await self.push_frame(
                    UserImageRequestFrame(self._participant_id), FrameDirection.UPSTREAM
                )
                self._push_time = time.time()
        await self.push_frame(frame, direction)


class UserImageRequestProcessor(UserImageBaseProcessor):
    """
    process:
    - request frame for push user image request frame to upstream
    - all frames to downstream
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if self._participant_id and isinstance(frame, self._request_frame_cls):
            await self.push_frame(
                UserImageRequestFrame(self._participant_id), FrameDirection.UPSTREAM
            )
        await self.push_frame(frame, direction)


class UserImageTextRequestProcessor(UserImageBaseProcessor):
    """
    process:
    - text frame in init_user_prompts for push user image request frame to upstream
    - user image raw frame to downstream
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if self._participant_id and isinstance(frame, TextFrame):
            if frame.text in self._init_user_prompts:
                await self.push_frame(
                    UserImageRequestFrame(self._participant_id), FrameDirection.UPSTREAM
                )
                if self._desc_img_prompt:
                    await self.push_frame(TextFrame(self._desc_img_prompt))
        elif isinstance(frame, UserImageRawFrame):
            await self.push_frame(frame)
        else:
            await self.push_frame(frame, direction)


class UserImageOrTextRequestProcessor(UserImageBaseProcessor):
    """
    process:
    - text frame in init_user_prompts for push user image request frame to upstream
    - all frames to downstream
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if self._participant_id and isinstance(frame, TextFrame):
            if self._init_user_prompts and frame.text in self._init_user_prompts:
                await self.push_frame(
                    UserImageRequestFrame(self._participant_id), FrameDirection.UPSTREAM
                )
        await self.push_frame(frame, direction)
