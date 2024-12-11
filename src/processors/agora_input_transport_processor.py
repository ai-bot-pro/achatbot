import logging
import asyncio
import time
from typing import Any

from apipeline.frames.base import Frame
from apipeline.frames.control_frames import StartFrame
from apipeline.processors.frame_processor import FrameDirection

from src.common.types import CHANNELS, RATE, AgoraParams
from src.processors.audio_input_processor import AudioVADInputProcessor
from src.services.agora_client import AgoraTransportClient
from src.types.frames.data_frames import (
    UserAudioRawFrame,
    UserImageRawFrame,
)
from src.types.frames.control_frames import UserImageRequestFrame
from src.types.frames.data_frames import AgoraTransportMessageFrame


class AgoraInputTransportProcessor(AudioVADInputProcessor):
    def __init__(self, client: AgoraTransportClient, params: AgoraParams, **kwargs):
        super().__init__(params, **kwargs)
        self._params = params
        self._client = client
        self._video_renderers = {}
        self._audio_in_task: asyncio.Task | None = None

    async def start(self, frame: StartFrame):
        # Parent start.
        await super().start(frame)
        # Join the room.
        await self._client.join()
        # Start sub room in audio stream task
        if not self._audio_in_task and (self._params.audio_in_enabled or self._params.vad_enabled):
            self._audio_in_task = asyncio.create_task(self._audio_in_task_handler())

    async def stop(self):
        # Cancel sub room in audio stream task
        if self._audio_in_task and (
            not self._audio_in_task.cancelled() or not self._audio_in_task.done()
        ):
            self._audio_in_task.cancel()
            await self._audio_in_task

        # Leave the room.
        await self._client.leave()
        # Parent stop.
        await super().stop()
        # Clear
        await self.cleanup()

    async def cleanup(self):
        await self._client.cleanup()
        await super().cleanup()

    #
    # FrameProcessor
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserImageRequestFrame):
            self.request_participant_image(frame.user_id)

    #
    # Frames
    #
    async def push_app_message(self, message: Any, sender: str):
        frame = AgoraTransportMessageFrame(message=message, participant_id=sender)
        await self.queue_frame(frame)

    #
    # Audio in
    #
    async def _audio_in_task_handler(self):
        logging.debug("Start sub room(channel) in audio stream task")
        while True:
            try:
                frame = await self._client.read_next_audio_frame()
                if frame:
                    await self.push_audio_frame(frame)
            except asyncio.CancelledError:
                logging.info("Cancelled sub room(channel) in audio stream task")
                break

    def capture_participant_audio(
        self,
        participant_id: str,
        sample_rate=None,
        num_channels=None,
    ):
        self._client.capture_participant_audio(
            participant_id,
            self._on_participant_audio_frame,
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

    async def _on_participant_audio_frame(self, frame: UserAudioRawFrame):
        """
        push to audio vad
        """
        if frame:
            # NOTE: use vad_audio_passthrough param to push queue with unblock
            await self.push_audio_frame(frame)

    #
    # Camera in
    #

    def capture_participant_video(
        self,
        participant_id: str,
        framerate: int = 30,
        video_source: str = "camera",
        color_format: str = "RGB",
    ):
        self._video_renderers[participant_id] = {
            "framerate": framerate,
            "timestamp": 0,
            "render_next_frame": False,
        }

        self._client.capture_participant_video(
            participant_id, self._on_participant_video_frame, framerate, video_source, color_format
        )

    def request_participant_image(self, participant_id: str):
        if participant_id in self._video_renderers:
            self._video_renderers[participant_id]["render_next_frame"] = True

    async def _on_participant_video_frame(self, frame: UserImageRawFrame):
        """
        control render image frame rate or request participant image to render
        """
        render_frame = False

        curr_time = time.time()
        prev_time = self._video_renderers[frame.user_id]["timestamp"] or curr_time
        framerate = self._video_renderers[frame.user_id]["framerate"]

        if framerate > 0:
            next_time = prev_time + 1 / framerate
            render_frame = (curr_time - next_time) < 0.1
        elif self._video_renderers[frame.user_id]["render_next_frame"]:
            # print("1_on_participant_video_frame->", self._video_renderers, frame)
            # e.g.: push UserImageRequestFrame to render a UserImageRawFrame
            self._video_renderers[frame.user_id]["render_next_frame"] = False
            render_frame = True

        if render_frame:
            await self.queue_frame(frame)

        self._video_renderers[frame.user_id]["timestamp"] = curr_time
