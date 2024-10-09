import logging
import asyncio
import time
from typing import Any

from apipeline.frames.base import Frame
from apipeline.frames.control_frames import StartFrame
from apipeline.processors.frame_processor import FrameDirection

from src.common.types import LivekitParams
from src.processors.audio_input_processor import AudioVADInputProcessor
from src.services.livekit_client import LivekitTransportClient
from src.types.frames.data_frames import (
    UserImageRawFrame,
)
from src.types.frames.control_frames import UserImageRequestFrame
from src.types.frames.data_frames import LivekitTransportMessageFrame


class LivekitInputTransportProcessor(AudioVADInputProcessor):

    def __init__(self, client: LivekitTransportClient, params: LivekitParams, **kwargs):
        super().__init__(params, **kwargs)
        self._params = params
        self._client = client
        self._video_renderers = {}
        self._audio_in_task = None
        self._camera_in_task = None

    async def start(self, frame: StartFrame):
        # Parent start.
        await super().start(frame)
        # Join the room.
        await self._client.join()
        # Create audio task. It reads audio frames from Daily and push them
        # internally for VAD processing.
        if self._params.audio_in_enabled or self._params.vad_enabled:
            self._audio_in_task = self.get_event_loop().create_task(self._audio_in_task_handler())

        if self._params.camera_in_enabled:
            self._camera_in_task = self.get_event_loop().create_task(self._camera_in_task_handler())

    async def stop(self):
        # Stop audio input thread.
        if self._params.audio_in_enabled or self._params.vad_enabled:
            self._audio_in_task.cancel()
            await self._audio_in_task
        # Stop camera input thread.
        if self._params.camera_in_enabled:
            self._camera_in_task.cancel()
            await self._camera_in_task
        # Leave the room.
        await self._client.leave()
        # Parent stop.
        await super().stop()

    async def cleanup(self):
        if self._params.audio_in_enabled or self._params.vad_enabled:
            self._audio_in_task.cancel()
            await self._audio_in_task
        if self._params.camera_in_enabled:
            self._camera_in_task.cancel()
            await self._camera_in_task
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
        frame = LivekitTransportMessageFrame(message=message, participant_id=sender)
        await self.queue_frame(frame)

    #
    # Audio in
    #

    async def _audio_in_task_handler(self):
        while True:
            try:
                frame = await self._client.read_next_audio_frame()
                if frame:
                    # NOTE: use vad_audio_passthrough param to push queue with unblock
                    # await self.push_frame(frame)
                    await self.push_audio_frame(frame)
            except asyncio.CancelledError:
                logging.info("Audio input task cancelled")
                break
            except Exception as e:
                logging.error(f"Error in audio input task: {e}")

    #
    # Camera in
    #
    async def _camera_in_task_handler(self):
        while True:
            try:
                frame = await self._client.read_next_image_frame(
                    target_color_mode=self._params.camera_in_color_format)
                if frame:
                    self._on_participant_video_frame(frame)
            except asyncio.CancelledError:
                logging.info("Video input task cancelled")
                break
            except Exception as e:
                logging.error(f"Error in video input task: {e}")

    def capture_participant_video(
            self,
            participant_id: str,
            framerate: int = 30,
            video_source: str = "camera",
            color_format: str = "RGB"):
        self._video_renderers[participant_id] = {
            "framerate": framerate,
            "timestamp": 0,
            "render_next_frame": False,
        }

        self._client.capture_participant_video(
            participant_id,
            self._on_participant_video_frame,
            framerate,
            video_source,
            color_format
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
        prev_time = self._video_renderers[frame.participant_id]["timestamp"] or curr_time
        framerate = self._video_renderers[frame.participant_id]["framerate"]

        if framerate > 0:
            next_time = prev_time + 1 / framerate
            render_frame = (curr_time - next_time) < 0.1
        elif self._video_renderers[frame.participant_id]["render_next_frame"]:
            # e.g.: push UserImageRequestFrame to render a UserImageRawFrame
            self._video_renderers[frame.participant_id]["render_next_frame"] = False
            render_frame = True

        if render_frame:
            await self.queue_frame(frame)

        self._video_renderers[frame.participant_id]["timestamp"] = curr_time
