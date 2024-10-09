import asyncio
import logging
from typing import Awaitable, Callable, List

import numpy as np
from PIL import Image
from pydantic import BaseModel
from apipeline.frames.data_frames import AudioRawFrame, ImageRawFrame

from src.common.types import SAMPLE_WIDTH, LivekitParams
from src.common.utils.audio_utils import convertSampleRateTo16khz
from src.types.frames.data_frames import LivekitTransportMessageFrame, TransportMessageFrame, UserImageRawFrame

try:
    from livekit import rtc, api
    from tenacity import retry, stop_after_attempt, wait_exponential
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use Livekit, you need to `pip install achatbot[livekit, livekit-api]`.")
    raise Exception(f"Missing module: {e}")


class LivekitCallbacks(BaseModel):
    on_connected: Callable[[], Awaitable[None]]
    on_disconnected: Callable[[], Awaitable[None]]
    on_participant_connected: Callable[[str], Awaitable[None]]
    on_participant_disconnected: Callable[[str], Awaitable[None]]
    on_audio_track_subscribed: Callable[[str], Awaitable[None]]
    on_audio_track_unsubscribed: Callable[[str], Awaitable[None]]
    on_video_track_subscribed: Callable[[str], Awaitable[None]]
    on_video_track_unsubscribed: Callable[[str], Awaitable[None]]
    on_data_received: Callable[[bytes, str], Awaitable[None]]
    on_first_participant_joined: Callable[[str], Awaitable[None]]


class LivekitTransportClient:
    def __init__(
        self,
        token: str,
        params: LivekitParams,
        callbacks: LivekitCallbacks,
        loop: asyncio.AbstractEventLoop,
    ):
        self._token = token
        self._token_claims: api.access_token.Claims | None = None
        if not self.verify_token():
            raise Exception(f"token {token} is not valid")

        self._room_name = self._token_claims.video.room
        self._params = params
        self._callbacks = callbacks
        self._loop = loop
        self._room = rtc.Room(loop=loop)
        self._participant_id: str = ""
        self._connected = False
        self._other_participant_has_joined = False

        self._audio_source: rtc.AudioSource | None = None
        self._audio_track: rtc.LocalAudioTrack | None = None
        self._audio_tracks = {}
        self._audio_queue = asyncio.Queue()

        self._video_source: rtc.VideoSource | None = None
        self._video_track: rtc.LocalVideoTrack | None = None
        self._video_tracks = {}
        self._video_queue = asyncio.Queue()
        self._start_capture_participant_video = False

        # Set up room event handlers to call register handlers
        # https://docs.livekit.io/home/client/events/#Events
        self._room.on("participant_connected")(self._on_participant_connected_wrapper)
        self._room.on("participant_disconnected")(self._on_participant_disconnected_wrapper)
        self._room.on("track_subscribed")(self._on_track_subscribed_wrapper)
        self._room.on("track_unsubscribed")(self._on_track_unsubscribed_wrapper)
        self._room.on("data_received")(self._on_data_received_wrapper)
        self._room.on("connected")(self._on_connected_wrapper)
        self._room.on("disconnected")(self._on_disconnected_wrapper)

    @property
    def participant_id(self) -> str:
        return self._participant_id

    def verify_token(self) -> bool:
        try:
            self._token_claims = api.TokenVerifier().verify(self._token)
        except Exception as e:
            logging.warning(f"verfiy {self._token} Exception: {e}")
            return False

        return True

    async def cleanup(self):
        await self.leave()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def join(self):
        if self._connected:
            return

        logging.info(f"Connecting to {self._room_name}")

        try:
            options = rtc.RoomOptions(auto_subscribe=True)
            if self._params.e2ee_shared_key:
                e2ee_options = rtc.E2EEOptions()
                e2ee_options.key_provider_options.shared_key = self._params.e2ee_shared_key
                options = rtc.RoomOptions(
                    auto_subscribe=True,
                    e2ee=e2ee_options,
                )

            await self._room.connect(
                self._params.websocket_url,
                self._token,
                options=options,
            )
            self._connected = True
            self._participant_id = self._room.local_participant.sid
            logging.info(f"Connected to {self._room_name}")

            # Set up audio source and track for pub out to microphone
            self._audio_source = rtc.AudioSource(
                self._params.audio_out_sample_rate, self._params.audio_out_channels
            )
            self._audio_track = rtc.LocalAudioTrack.create_audio_track(
                "achatbot-audio", self._audio_source
            )
            options = rtc.TrackPublishOptions()
            options.source = rtc.TrackSource.SOURCE_MICROPHONE
            publication = await self._room.local_participant.publish_track(
                self._audio_track, options)
            logging.info(f"audio track local publication {publication}")

            # Set up video source and track for pub out to camera
            self._video_source = rtc.VideoSource(
                self._params.camera_out_width,
                self._params.camera_out_height,
            )
            self._video_track = rtc.LocalVideoTrack.create_video_track(
                "achatbot-video", self._video_source)
            options = rtc.TrackPublishOptions()
            options.source = rtc.TrackSource.SOURCE_CAMERA
            publication = await self._room.local_participant.publish_track(
                self._video_track, options)
            logging.info(f"video track local publication {publication}")

            await self._callbacks.on_connected()

            # Check if there are already participants in the room
            participants = self.get_participants()
            if participants and not self._other_participant_has_joined:
                self._other_participant_has_joined = True
                await self._callbacks.on_first_participant_joined(participants[0])
        except Exception as e:
            logging.error(f"Error connecting to {self._room_name}: {e}")
            raise

    async def leave(self):
        if not self._connected:
            return

        logging.info(f"Disconnecting from {self._room_name}")
        await self._room.disconnect()
        self._connected = False
        logging.info(f"Disconnected from {self._room_name}")
        await self._callbacks.on_disconnected()

    async def send_message(self, frame: TransportMessageFrame):
        if not self._connected:
            return

        destination_identities = []
        if isinstance(frame, LivekitTransportMessageFrame):
            destination_identities.append(frame.participant_id)
        data = frame.message.encode()

        try:
            await self._room.local_participant.publish_data(
                data,
                reliable=True,
                destination_identities=destination_identities,
            )
        except Exception as e:
            logging.error(f"Error sending data: {e}")

    def get_participants(self) -> List[str]:
        return [p.sid for p in self._room.remote_participants.values()]

    async def get_participant_metadata(self, participant_id: str) -> dict:
        participant = self._room.remote_participants.get(participant_id)
        if participant:
            return {
                "id": participant.sid,
                "name": participant.name,
                "metadata": participant.metadata,
                "is_speaking": participant.is_speaking,
            }
        return {}

    async def set_participant_metadata(self, metadata: str):
        await self._room.local_participant.set_metadata(metadata)

    async def mute_participant(self, participant_id: str):
        participant = self._room.remote_participants.get(participant_id)
        if participant:
            for track in participant.tracks.values():
                if track.kind == "audio":
                    await track.set_enabled(False)

    async def unmute_participant(self, participant_id: str):
        participant = self._room.remote_participants.get(participant_id)
        if participant:
            for track in participant.tracks.values():
                if track.kind == "audio":
                    await track.set_enabled(True)

    # Wrapper methods for event handlers
    def _on_participant_connected_wrapper(self, participant: rtc.RemoteParticipant):
        asyncio.create_task(self._async_on_participant_connected(participant))

    def _on_participant_disconnected_wrapper(self, participant: rtc.RemoteParticipant):
        asyncio.create_task(self._async_on_participant_disconnected(participant))

    def _on_track_subscribed_wrapper(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        asyncio.create_task(self._async_on_track_subscribed(track, publication, participant))

    def _on_track_unsubscribed_wrapper(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        asyncio.create_task(self._async_on_track_unsubscribed(track, publication, participant))

    def _on_data_received_wrapper(self, data: rtc.DataPacket):
        asyncio.create_task(self._async_on_data_received(data))

    def _on_connected_wrapper(self):
        asyncio.create_task(self._async_on_connected())

    def _on_disconnected_wrapper(self):
        asyncio.create_task(self._async_on_disconnected())

    # Async methods for event handling
    async def _async_on_participant_connected(self, participant: rtc.RemoteParticipant):
        logging.info(f"Participant connected: {participant.identity}")
        await self._callbacks.on_participant_connected(participant.sid)
        if not self._other_participant_has_joined:
            self._other_participant_has_joined = True
            await self._callbacks.on_first_participant_joined(participant.sid)

    async def _async_on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        logging.info(f"Participant disconnected: {participant.identity}")
        await self._callbacks.on_participant_disconnected(participant.sid)
        if len(self.get_participants()) == 0:
            self._other_participant_has_joined = False

    async def _async_on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if track.kind == rtc.TrackKind.KIND_AUDIO \
                and self._params.audio_in_enabled or self._params.vad_enabled:
            logging.info(f"Audio track subscribed: {track} from participant {participant.sid}")
            self._audio_tracks[participant.sid] = track
            audio_stream = rtc.AudioStream(track)
            asyncio.create_task(self._process_audio_stream(audio_stream, participant.sid))
            await self._callbacks.on_audio_track_subscribed(participant.sid)
        elif track.kind == rtc.TrackKind.KIND_VIDEO \
                and self._params.camera_in_enabled \
                and self._start_capture_participant_video:
            logging.info(f"Video track subscribed: {track} from participant {participant.sid}")
            self._video_tracks[participant.sid] = track
            video_stream = rtc.VideoStream(track)
            asyncio.create_task(self._process_video_stream(video_stream, participant.sid))
            await self._callbacks.on_video_track_subscribed(participant.sid)

    async def _async_on_track_unsubscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logging.info(f"Track unsubscribed: {publication.sid} from {participant.identity}")
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            await self._callbacks.on_audio_track_unsubscribed(participant.sid)
        elif track.kind == rtc.TrackKind.KIND_VIDEO:
            self._start_capture_participant_video = False
            await self._callbacks.on_video_track_unsubscribed(participant.sid)

    async def _async_on_data_received(self, data: rtc.DataPacket):
        await self._callbacks.on_data_received(data.data, data.participant.sid)

    async def _async_on_connected(self):
        await self._callbacks.on_connected()

    async def _async_on_disconnected(self, reason=None):
        self._connected = False
        logging.info(f"Disconnected from {self._room_name}. Reason: {reason}")
        await self._callbacks.on_disconnected()

    # Audio in

    async def _process_audio_stream(self, audio_stream: rtc.AudioStream, participant_id: str):
        logging.info(f"Started processing audio stream for participant {participant_id}")
        async for event in audio_stream:
            if isinstance(event, rtc.AudioFrameEvent):
                await self._audio_queue.put((event, participant_id))
            else:
                logging.warning(f"Received unexpected event type: {type(event)}")

    async def read_next_audio_frame(self) -> AudioRawFrame | None:
        audio_frame_event, _ = await self._audio_queue.get()
        audio_frame = self._convert_input_audio(audio_frame_event)
        return audio_frame

    def _convert_input_audio(
        self, audio_frame_event: rtc.AudioFrameEvent
    ) -> AudioRawFrame | None:
        audio_data = audio_frame_event.frame.data
        original_sample_rate = audio_frame_event.frame.sample_rate
        original_num_channels = audio_frame_event.frame.num_channels

        # Allow 8kHz and 16kHz, other sampple rate convert to 16kHz
        if original_sample_rate not in [8000, 16000]:
            audio_data = convertSampleRateTo16khz(audio_data, original_sample_rate)
            sample_rate = 16000
        else:
            sample_rate = original_sample_rate

        if sample_rate != self._params.audio_in_sample_rate:
            self._params.audio_in_sample_rate = sample_rate
            if self._params.vad_enabled and self._params.vad_analyzer:
                self._params.vad_analyzer.set_args({
                    "sample_rate": sample_rate,
                    "num_channels": original_num_channels,
                })

        return AudioRawFrame(
            audio=audio_data,
            sample_rate=sample_rate,
            num_channels=original_num_channels,
        )

    # Audio out

    async def write_raw_audio_frames(self, frames: bytes,):
        if not self._connected or not self._audio_source:
            return

        try:
            audio_frame: rtc.AudioFrame = self._convert_output_audio(frames)
            await self._audio_source.capture_frame(audio_frame)
        except Exception as e:
            logging.error(f"Error publishing audio: {e}")

    def _convert_output_audio(self, audio_data: bytes) -> rtc.AudioFrame:
        bytes_per_sample = SAMPLE_WIDTH  # Assuming 16-bit audio sample_width
        total_samples = len(audio_data) // bytes_per_sample
        samples_per_channel = total_samples // self._params.audio_out_channels

        return rtc.AudioFrame(
            data=audio_data,
            sample_rate=self._params.audio_out_sample_rate,
            num_channels=self._params.audio_out_channels,
            samples_per_channel=samples_per_channel,
        )

    # Camera in

    async def _process_video_stream(self, video_stream: rtc.VideoStream, participant_id: str):
        logging.info(f"Started processing video stream for participant {participant_id}")
        async for event in video_stream:
            if isinstance(event, rtc.VideoFrameEvent):
                await self._video_queue.put((event, participant_id))
            else:
                logging.warning(f"Received unexpected event type: {type(event)}")

    async def read_next_image_frame(
            self, target_color_mode: str = "RGB") -> UserImageRawFrame | None:
        video_frame_event, participant_id = await self._video_queue.get()
        image_frame = self._convert_input_video_image(
            participant_id, video_frame_event,
            target_color_mode=target_color_mode)
        return image_frame

    def _convert_input_video_image(
            self,
            participant_id: str,
            video_frame_event: rtc.VideoFrameEvent,
            target_color_mode: str = "RGB") -> UserImageRawFrame | None:
        buffer = video_frame_event.frame

        match buffer.type:
            case rtc.VideoBufferType.RGBA:
                image = Image.frombuffer(
                    "RGBA", (buffer.width, buffer.height), buffer.data, "raw", "RGBA", 0, 1)
            case rtc.VideoBufferType.ABGR:
                image = Image.frombuffer(
                    "ABGR", (buffer.width, buffer.height), buffer.data, "raw", "ABGR", 0, 1)
            case rtc.VideoBufferType.ARGB:
                image = Image.frombuffer(
                    "ARGB", (buffer.width, buffer.height), buffer.data, "raw", "ARGB", 0, 1)
            case rtc.VideoBufferType.BGRA:
                image = Image.frombuffer(
                    "BGRA", (buffer.width, buffer.height), buffer.data, "raw", "BGRA", 0, 1)
            case rtc.VideoBufferType.RGB24:
                image = Image.frombuffer(
                    "RGB", (buffer.width, buffer.height), buffer.data, "raw", "RGB", 0, 1)
            case _:
                logging.warning(f"buffer type:{buffer.type} un support convert")
                return None

        image = image.convert(target_color_mode)
        return UserImageRawFrame(
            image=image.tobytes(),
            size=(buffer.width, buffer.height),
            mode=target_color_mode,
            format="JPEG",
            participant_id=participant_id,
        )

    def capture_participant_video(
            self,
            participant_id: str,
            callback: Callable,
            framerate: int = 30,
            video_source: str = "camera",
            color_format: str = "RGB"):
        # just open to capture_participant_video
        self._start_capture_participant_video = True

    # Camera out

    async def write_frame_to_camera(self, frame: ImageRawFrame):
        """
        publish image frame  to camera
        !NOTE: (now just support RGB color_format/mode)
        !TODO: from diff detector stream to display, like audio out sample rate to play
        """
        rtc_frame_type = rtc.VideoBufferType.RGB24
        match frame.mode:
            case "RGB":
                rtc_frame_type = rtc.VideoBufferType.RGB24
            case _:
                logging.warning(f"frame mode:{frame.mode} unsupport pub to camera")
                return
        video_frame = rtc.VideoFrame(
            self._params.camera_out_width,
            self._params.camera_out_height,
            rtc_frame_type,
            frame.image
        )
        self._video_source.capture_frame(video_frame)
        pass
