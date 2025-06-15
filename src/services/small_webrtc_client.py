import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional

import numpy as np
from pydantic import BaseModel

from src.common.types import AudioCameraParams
from src.services.webrtc_peer_connection import (
    SmallWebRTCConnection,
    RawAudioTrack,
    RawVideoTrack,
    SmallWebRTCTrack,
)
from src.types.frames import (
    UserAudioRawFrame,
    UserImageRawFrame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    TransportMessageFrame,
)

try:
    import cv2
    from aiortc import VideoStreamTrack
    from aiortc.mediastreams import AudioStreamTrack, MediaStreamError
    from av import AudioFrame, AudioResampler, VideoFrame
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error("In order to use the SmallWebRTC, you need to `pip install achatbot[webrtc]`.")
    raise Exception(f"Missing module: {e}")


class SmallWebRTCCallbacks(BaseModel):
    on_app_message: Callable[[SmallWebRTCConnection, Any], Awaitable[None]]
    on_client_connected: Callable[[SmallWebRTCConnection], Awaitable[None]]
    on_client_disconnected: Callable[[SmallWebRTCConnection], Awaitable[None]]


class SmallWebRTCClient:
    FORMAT_CONVERSIONS = {
        "yuv420p": cv2.COLOR_YUV2RGB_I420,
        "yuvj420p": cv2.COLOR_YUV2RGB_I420,  # OpenCV treats both the same
        "nv12": cv2.COLOR_YUV2RGB_NV12,
        "gray": cv2.COLOR_GRAY2RGB,
    }

    def __init__(self, webrtc_connection: SmallWebRTCConnection, callbacks: SmallWebRTCCallbacks):
        self._webrtc_connection = webrtc_connection
        self._closing = False
        self._callbacks = callbacks

        self._audio_output_track = None
        self._video_output_track = None
        self._audio_input_track: Optional[SmallWebRTCTrack] = None
        self._video_input_track: Optional[SmallWebRTCTrack] = None

        self._params = None
        self._audio_in_channels = None
        self._in_sample_rate = None
        self._out_sample_rate = None

        # We are always resampling it for 16000 if the sample_rate that we receive is bigger than that.
        # otherwise we face issues with Silero VAD
        self._resampler = AudioResampler("s16", "mono", 16000)

        # connect only once
        self._connecting = False

        @self._webrtc_connection.event_handler("connected")
        async def on_connected(connection: SmallWebRTCConnection):
            logging.debug("Peer connection established.")
            await self._handle_client_connected()

        @self._webrtc_connection.event_handler("disconnected")
        async def on_disconnected(connection: SmallWebRTCConnection):
            logging.debug("Peer connection lost.")
            await self._handle_peer_disconnected()

        @self._webrtc_connection.event_handler("closed")
        async def on_closed(connection: SmallWebRTCConnection):
            logging.debug("Client connection closed.")
            await self._handle_client_closed()

        @self._webrtc_connection.event_handler("app-message")
        async def on_app_message(connection: SmallWebRTCConnection, message: Any):
            await self._handle_app_message(message)

    def _convert_frame(self, frame_array: np.ndarray, format_name: str) -> np.ndarray:
        """Convert a given frame to RGB format based on the input format.

        Args:
            frame_array (np.ndarray): The input frame.
            format_name (str): The format of the input frame.

        Returns:
            np.ndarray: The converted RGB frame.

        Raises:
            ValueError: If the format is unsupported.
        """
        if format_name.startswith("rgb"):  # Already in RGB, no conversion needed
            return frame_array

        conversion_code = SmallWebRTCClient.FORMAT_CONVERSIONS.get(format_name)

        if conversion_code is None:
            raise ValueError(f"Unsupported format: {format_name}")

        return cv2.cvtColor(frame_array, conversion_code)

    async def read_video_frame(self):
        """Reads a video frame from the given MediaStreamTrack, converts it to RGB,
        and creates an UserImageRawFrame.
        """
        while True:
            if self._video_input_track is None:
                await asyncio.sleep(0.01)
                continue

            try:
                frame = await asyncio.wait_for(self._video_input_track.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                if self._webrtc_connection.is_connected():
                    logging.warning("Timeout: No video frame received within the specified time.")
                    # self._webrtc_connection.ask_to_renegotiate()
                frame = None
            except MediaStreamError:
                logging.warning(
                    "Received an unexpected media stream error while reading the audio."
                )
                frame = None

            if frame is None or not isinstance(frame, VideoFrame):
                # If no valid frame, sleep for a bit
                await asyncio.sleep(0.01)
                continue

            format_name = frame.format.name
            # Convert frame to NumPy array in its native format
            frame_array = frame.to_ndarray(format=format_name)
            frame_rgb = self._convert_frame(frame_array, format_name)

            image_frame = UserImageRawFrame(
                user_id=self._webrtc_connection.pc_id,
                image=frame_rgb.tobytes(),
                size=(frame.width, frame.height),
                format="RGB",
            )

            yield image_frame

    async def read_audio_frame(self):
        """Reads 20ms of audio from the given MediaStreamTrack and creates an InputAudioRawFrame."""
        while True:
            if self._audio_input_track is None:
                await asyncio.sleep(0.01)
                continue

            try:
                frame = await asyncio.wait_for(self._audio_input_track.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                if self._webrtc_connection.is_connected():
                    logging.warning("Timeout: No audio frame received within the specified time.")
                frame = None
            except MediaStreamError:
                logging.warning(
                    "Received an unexpected media stream error while reading the audio."
                )
                frame = None

            if frame is None or not isinstance(frame, AudioFrame):
                # If we don't read any audio let's sleep for a little bit (i.e. busy wait).
                await asyncio.sleep(0.01)
                continue

            if frame.sample_rate > self._in_sample_rate:
                resampled_frames = self._resampler.resample(frame)
                for resampled_frame in resampled_frames:
                    # 16-bit PCM bytes
                    pcm_bytes = resampled_frame.to_ndarray().astype(np.int16).tobytes()
                    audio_frame = UserAudioRawFrame(
                        user_id=self._webrtc_connection.pc_id,
                        audio=pcm_bytes,
                        sample_rate=resampled_frame.sample_rate,
                        num_channels=self._audio_in_channels,
                    )
                    yield audio_frame
            else:
                # 16-bit PCM bytes
                pcm_bytes = frame.to_ndarray().astype(np.int16).tobytes()
                audio_frame = UserAudioRawFrame(
                    user_id=self._webrtc_connection.pc_id,
                    audio=pcm_bytes,
                    sample_rate=frame.sample_rate,
                    num_channels=self._audio_in_channels,
                )
                yield audio_frame

    async def write_audio_frame(self, frame: OutputAudioRawFrame):
        if self._can_send() and self._audio_output_track:
            await self._audio_output_track.add_audio_bytes(frame.audio)

    async def write_raw_audio_frames(self, frames: bytes):
        if self._can_send() and self._audio_output_track:
            await self._audio_output_track.add_audio_bytes(frames)

    async def write_video_frame(self, frame: OutputImageRawFrame):
        if self._can_send() and self._video_output_track:
            self._video_output_track.add_video_frame(frame)

    async def setup(self, _params: AudioCameraParams, frame):
        self._audio_in_channels = _params.audio_in_channels
        self._in_sample_rate = _params.audio_in_sample_rate or frame.audio_in_sample_rate
        self._out_sample_rate = _params.audio_out_sample_rate or frame.audio_out_sample_rate
        self._params = _params

    async def connect(self):
        if self._webrtc_connection.is_connected():
            # already initialized
            logging.info(f"Small WebRTC already initialized")
            return
        if self._connecting is True:
            # connecting
            logging.info(f"Small WebRTC is connecting")
            return

        self._connecting = True
        logging.info(
            f"Connecting to Small WebRTC connectionState:{self._webrtc_connection.connectionState}"
        )
        await self._webrtc_connection.connect()

    async def disconnect(self):
        if self.is_connected and not self.is_closing:
            logging.info(f"Disconnecting to Small WebRTC")
            self._closing = True
            self._connecting = False
            await self._webrtc_connection.disconnect()
            await self._handle_peer_disconnected()
            logging.info(f"Disconnected to Small WebRTC")

    async def send_message(self, frame: TransportMessageFrame):
        if self._can_send():
            self._webrtc_connection.send_app_message(frame.message)

    async def _handle_client_connected(self):
        # There is nothing to do here yet, the pipeline is still not ready
        if not self._params:
            return

        self._audio_input_track = self._webrtc_connection.audio_input_track()
        self._video_input_track = self._webrtc_connection.video_input_track()
        if self._params.audio_out_enabled:
            self._audio_output_track = RawAudioTrack(sample_rate=self._out_sample_rate)
            self._webrtc_connection.replace_audio_track(self._audio_output_track)

        if self._params.camera_out_enabled:
            self._video_output_track = RawVideoTrack(
                width=self._params.camera_out_width, height=self._params.camera_out_height
            )
            self._webrtc_connection.replace_video_track(self._video_output_track)

        await self._callbacks.on_client_connected(self._webrtc_connection)

    async def _handle_peer_disconnected(self):
        self._audio_input_track = None
        self._video_input_track = None
        self._audio_output_track = None
        self._video_output_track = None

    async def _handle_client_closed(self):
        self._audio_input_track = None
        self._video_input_track = None
        self._audio_output_track = None
        self._video_output_track = None
        await self._callbacks.on_client_disconnected(self._webrtc_connection)

    async def _handle_app_message(self, message: Any):
        await self._callbacks.on_app_message(self._webrtc_connection, message)

    def _can_send(self):
        return self.is_connected and not self.is_closing

    @property
    def is_connected(self) -> bool:
        return self._webrtc_connection.is_connected()

    @property
    def is_closing(self) -> bool:
        return self._closing
