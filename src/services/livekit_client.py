import asyncio
import logging
import os
from typing import Awaitable, Callable, List

from pydantic import BaseModel
import numpy as np
from apipeline.frames.data_frames import AudioRawFrame, ImageRawFrame

from src.common.types import SAMPLE_WIDTH, LiveKitParams
from src.common.utils.audio_utils import convertSampleRateTo16khz
from src.types.frames.data_frames import LivekitTransportMessageFrame, TransportMessageFrame

try:
    from livekit import rtc
    from tenacity import retry, stop_after_attempt, wait_exponential
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error("In order to use LiveKit, you need to `pip install achatbot[livekit]`.")
    raise Exception(f"Missing module: {e}")


class LivekitCallbacks(BaseModel):
    on_connected: Callable[[], Awaitable[None]]
    on_disconnected: Callable[[], Awaitable[None]]
    on_participant_connected: Callable[[str], Awaitable[None]]
    on_participant_disconnected: Callable[[str], Awaitable[None]]
    on_audio_track_subscribed: Callable[[str], Awaitable[None]]
    on_audio_track_unsubscribed: Callable[[str], Awaitable[None]]
    on_data_received: Callable[[bytes, str], Awaitable[None]]
    on_first_participant_joined: Callable[[str], Awaitable[None]]


class LivekitTransportClient:
    def __init__(
        self,
        websocket_url: str,
        token: str,
        room_name: str,
        params: LiveKitParams,
        callbacks: LivekitCallbacks,
        loop: asyncio.AbstractEventLoop,
    ):
        self._websocket_url = websocket_url
        self._token = token
        self._room_name = room_name
        self._params = params
        self._callbacks = callbacks
        self._loop = loop
        self._room = rtc.Room(loop=loop)
        self._participant_id: str = ""
        self._connected = False
        self._audio_source: rtc.AudioSource | None = None
        self._audio_track: rtc.LocalAudioTrack | None = None
        self._audio_tracks = {}
        self._audio_queue = asyncio.Queue()
        self._other_participant_has_joined = False

        # Set up room event handlers
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def join(self):
        if self._connected:
            return

        logging.info(f"Connecting to {self._room_name}")

        try:
            await self._room.connect(
                self._websocket_url,
                self._token,
                options=rtc.RoomOptions(auto_subscribe=True),
            )
            self._connected = True
            self._participant_id = self._room.local_participant.sid
            logging.info(f"Connected to {self._room_name}")

            # Set up audio source and track
            self._audio_source = rtc.AudioSource(
                self._params.audio_out_sample_rate, self._params.audio_out_channels
            )
            self._audio_track = rtc.LocalAudioTrack.create_audio_track(
                "achatbot-audio", self._audio_source
            )
            options = rtc.TrackPublishOptions()
            options.source = rtc.TrackSource.SOURCE_MICROPHONE
            await self._room.local_participant.publish_track(self._audio_track, options)

            # TODO: sub/pub participant Video and Images

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
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logging.info(f"Audio track subscribed: {track.sid} from participant {participant.sid}")
            self._audio_tracks[participant.sid] = track
            audio_stream = rtc.AudioStream(track)
            asyncio.create_task(self._process_audio_stream(audio_stream, participant.sid))

    async def _async_on_track_unsubscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logging.info(f"Track unsubscribed: {publication.sid} from {participant.identity}")
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            await self._callbacks.on_audio_track_unsubscribed(participant.sid)

    async def _async_on_data_received(self, data: rtc.DataPacket):
        await self._callbacks.on_data_received(data.data, data.participant.sid)

    async def _async_on_connected(self):
        await self._callbacks.on_connected()

    async def _async_on_disconnected(self, reason=None):
        self._connected = False
        logging.info(f"Disconnected from {self._room_name}. Reason: {reason}")
        await self._callbacks.on_disconnected()

    async def _process_audio_stream(self, audio_stream: rtc.AudioStream, participant_id: str):
        logging.info(f"Started processing audio stream for participant {participant_id}")
        async for event in audio_stream:
            if isinstance(event, rtc.AudioFrameEvent):
                await self._audio_queue.put((event, participant_id))
            else:
                logging.warning(f"Received unexpected event type: {type(event)}")

    async def cleanup(self):
        await self.leave()

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

    # Camera

    def capture_participant_video(
            self,
            participant_id: str,
            callback: Callable,
            framerate: int = 30,
            video_source: str = "camera",
            color_format: str = "RGB"):
        # TODO: sub participant_id's video_source to do callback
        pass

    async def write_frame_to_camera(self, frame: ImageRawFrame):
        # TODO: pub image frame
        pass
