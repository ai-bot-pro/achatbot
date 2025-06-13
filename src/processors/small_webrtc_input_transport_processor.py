import logging
from typing import Any


from apipeline.frames import Frame, StartFrame, EndFrame, CancelFrame
from apipeline.processors.frame_processor import FrameDirection

from src.types.frames import TransportMessageFrame, UserImageRawFrame, UserImageRequestFrame
from src.common.types import AudioCameraParams
from src.services.small_webrtc_client import SmallWebRTCClient
from src.processors.audio_input_processor import AudioVADInputProcessor


class SmallWebRTCInputProcessor(AudioVADInputProcessor):
    def __init__(
        self,
        client: SmallWebRTCClient,
        params: AudioCameraParams,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._client = client
        self._params = params
        self._receive_audio_task = None
        self._receive_video_task = None
        self._image_requests = {}

        # Whether we have seen a StartFrame already.
        self._initialized = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserImageRequestFrame):
            await self.request_participant_image(frame)

    async def start(self, frame: StartFrame):
        await super().start(frame)

        if self._initialized:
            return

        self._initialized = True

        await self._client.setup(self._params, frame)
        await self._client.connect()
        if not self._receive_audio_task and self._params.audio_in_enabled:
            self._receive_audio_task = self.create_task(self._receive_audio())
        if not self._receive_video_task and self._params.video_in_enabled:
            self._receive_video_task = self.create_task(self._receive_video())
        await self.set_transport_ready(frame)

    async def _stop_tasks(self):
        if self._receive_audio_task:
            await self.cancel_task(self._receive_audio_task)
            self._receive_audio_task = None
        if self._receive_video_task:
            await self.cancel_task(self._receive_video_task)
            self._receive_video_task = None

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._stop_tasks()
        await self._client.disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._stop_tasks()
        await self._client.disconnect()

    async def _receive_audio(self):
        try:
            async for audio_frame in self._client.read_audio_frame():
                if audio_frame:
                    await self.push_audio_frame(audio_frame)

        except Exception as e:
            logging.error(f"{self} exception receiving data: {e.__class__.__name__} ({e})")

    async def _receive_video(self):
        try:
            async for video_frame in self._client.read_video_frame():
                if video_frame:
                    await self.push_video_frame(video_frame)

                    # Check if there are any pending image requests and create UserImageRawFrame
                    if self._image_requests:
                        for req_id, request_frame in list(self._image_requests.items()):
                            # Create UserImageRawFrame using the current video frame
                            image_frame = UserImageRawFrame(
                                user_id=request_frame.user_id,
                                request=request_frame,
                                image=video_frame.image,
                                size=video_frame.size,
                                format=video_frame.format,
                            )
                            # Push the frame to the pipeline
                            await self.push_video_frame(image_frame)
                            # Remove from pending requests
                            del self._image_requests[req_id]

        except Exception as e:
            logging.error(f"{self} exception receiving data: {e.__class__.__name__} ({e})")

    async def push_app_message(self, message: Any):
        logging.debug(f"Received app message inside SmallWebRTCInputTransport  {message}")
        frame = TransportMessageFrame(message=message, urgent=True)
        await self.push_frame(frame)

    # Add this method similar to DailyInputTransport.request_participant_image
    async def request_participant_image(self, frame: UserImageRequestFrame):
        """Requests an image frame from the participant's video stream.

        When a UserImageRequestFrame is received, this method will store the request
        and the next video frame received will be converted to a UserImageRawFrame.
        """
        logging.debug(f"Requesting image from participant: {frame.user_id}")

        # Store the request
        request_id = f"{frame.function_name}:{frame.tool_call_id}"
        self._image_requests[request_id] = frame

        # If we're not already receiving video, try to get a frame now
        if not self._receive_video_task and self._params.video_in_enabled:
            # Start video reception if it's not already running
            self._receive_video_task = self.create_task(self._receive_video())
