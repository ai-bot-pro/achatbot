import asyncio
import logging
import pickle
from typing import AsyncGenerator, Awaitable, Callable, Dict, List

import numpy as np
from PIL import Image
from pydantic import BaseModel
from apipeline.frames.data_frames import AudioRawFrame, ImageRawFrame

from src.common.types import SAMPLE_WIDTH, LivekitParams
from src.common.utils.audio_utils import resample_audio
from src.types.frames.data_frames import (
    LivekitTransportMessageFrame,
    TransportMessageFrame,
    UserAudioRawFrame,
    UserImageRawFrame,
)

try:
    from livekit import rtc, api, protocol
    # from tenacity import retry, stop_after_attempt, wait_exponential
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use Livekit, you need to `pip install achatbot[livekit, livekit-api]`."
    )
    raise Exception(f"Missing module: {e}")


class LivekitCallbacks(BaseModel):
    on_connected: Callable[[rtc.Room], Awaitable[None]]
    on_error: Callable[[str], Awaitable[None]]
    on_connection_state_changed: Callable[[rtc.ConnectionState], Awaitable[None]]
    # on_disconnected: Callable[[Union[protocol.models.DisconnectReason, str]], Awaitable[None]]
    on_disconnected: Callable[[str], Awaitable[None]]  # use str replace Union for python3.10
    on_participant_connected: Callable[[rtc.RemoteParticipant], Awaitable[None]]
    on_participant_disconnected: Callable[[rtc.RemoteParticipant], Awaitable[None]]
    on_audio_track_subscribed: Callable[[rtc.RemoteParticipant], Awaitable[None]]
    on_audio_track_unsubscribed: Callable[[rtc.RemoteParticipant], Awaitable[None]]
    on_video_track_subscribed: Callable[[rtc.RemoteParticipant], Awaitable[None]]
    on_video_track_unsubscribed: Callable[[rtc.RemoteParticipant], Awaitable[None]]
    on_data_received: Callable[[bytes, rtc.RemoteParticipant], Awaitable[None]]
    on_first_participant_joined: Callable[[rtc.RemoteParticipant], Awaitable[None]]


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
        self._join_name = (
            self._token_claims.name if self._token_claims.name else self._token_claims.identity
        )
        self._local_participant: rtc.Participant | None = None  # room VirtualDevice bot
        self._params = params
        self._callbacks = callbacks
        self._loop = loop
        self._room = rtc.Room(loop=loop)

        self._joined = False
        self._joining = False
        self._leaving = False

        self._other_participant_has_joined = False
        self._current_sub_audio_participant_id: str = ""
        self._current_sub_video_participant_id: str = ""

        # audio out
        # local participant audio stream
        self._out_audio_source: rtc.AudioSource | None = None
        self._out_audio_track: rtc.LocalAudioTrack | None = None
        # audio in
        # TODO:local participant audio stream unsupport to get from local speaker
        self._in_local_audio_stream: rtc.AudioStream | None = None
        # for sub multi remote participant audio stream,
        self._in_participant_audio_tracks: Dict[str, rtc.AudioTrack] = {}
        self._on_participant_audio_frame_task: asyncio.Task | None = None
        self._capture_participant_audio_stream: rtc.AudioStream | None = None
        # TODO: switch room anchor participant, need know room anchor
        self._participant_audio_stream: rtc.AudioStream | None = None
        self._in_audio_queue = asyncio.Queue()
        self._in_audio_task: asyncio.Task | None = None

        # video out
        # local participant video stream
        self._out_video_source: rtc.VideoSource | None = None
        self._out_video_track: rtc.LocalVideoTrack | None = None
        # video in
        # for sub multi remote participant video track,
        # chat bot need see participant video, so need sub participant video stream
        self._in_participant_video_tracks: Dict[str, rtc.VideoTrack] = {}
        self._on_participant_video_frame_task: asyncio.Task | None = None
        self._capture_participant_video_stream: rtc.VideoStream | None = None

        # Set up room event handlers to call register handlers
        # https://docs.livekit.io/home/client/events/#Events
        # NOTE!!!! please see Room on method Available events Arguments :)
        self._room.on("participant_connected")(self._on_participant_connected_wrapper)
        self._room.on("participant_disconnected")(self._on_participant_disconnected_wrapper)
        self._room.on("track_subscribed")(self._on_track_subscribed_wrapper)
        self._room.on("track_unsubscribed")(self._on_track_unsubscribed_wrapper)
        self._room.on("data_received")(self._on_data_received_wrapper)
        self._room.on("connected")(self._on_connected_wrapper)
        self._room.on("disconnected")(self._on_disconnected_wrapper)
        self._room.on("connection_state_changed")(self._on_connection_state_changed_wrapper)

    @property
    def participant_id(self) -> str:
        if self._local_participant:
            return self._local_participant.sid
        return ""

    @property
    def local_participant(self) -> rtc.Participant | None:
        return self._local_participant

    def verify_token(self) -> bool:
        try:
            self._token_claims = api.TokenVerifier().verify(self._token)
        except Exception as e:
            logging.warning(f"verfiy {self._token} Exception: {e}")
            return False

        return True

    async def cleanup(self):
        if self._in_audio_task and not self._in_audio_task.cancelled():
            self._in_audio_task.cancel()
            await self._in_audio_task
        if (
            self._on_participant_audio_frame_task
            and not self._on_participant_audio_frame_task.cancelled()
        ):
            self._on_participant_audio_frame_task.cancel()
            await self._on_participant_audio_frame_task
        if (
            self._on_participant_video_frame_task
            and not self._on_participant_video_frame_task.cancelled()
        ):
            self._on_participant_video_frame_task.cancel()
            await self._on_participant_video_frame_task

    # @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _join(self):
        self._joining = True
        participants = self.get_participants()
        logging.info(
            f"{self._join_name} Connecting to {self._room_name},"
            f" current remote participants:{participants}"
        )

        options = rtc.RoomOptions(auto_subscribe=True)
        if self._params.e2ee_shared_key:
            e2ee_options = rtc.E2EEOptions()
            e2ee_options.key_provider_options.shared_key = self._params.e2ee_shared_key
            options = rtc.RoomOptions(
                auto_subscribe=True,
                e2ee=e2ee_options,
            )

        try:
            await self._room.connect(
                self._params.websocket_url,
                self._token,
                options=options,
            )
        except rtc.ConnectError as e:
            self._joining = False
            self._joined = False
            raise e

        logging.info(
            f"local_participant:{self._room.local_participant} joined room: {self._room_name} connection_state: {self._room.connection_state}"
        )
        self._joining = False
        self._joined = True
        self._local_participant = self._room.local_participant
        await self._callbacks.on_connected(self._room)

    async def join(self):
        # input and output procssor maybe do join room at the same time
        # need check join state, avoid duplicate join
        if self._joined or self._joining:
            logging.warning(f"{self._join_name} has connected {self._room_name}")
            return

        try:
            await self._join()

            # TODO: transcription_enabled to start

            # Set up audio source and track
            # - for pub/write out to local_participant microphone
            if self._params.audio_out_enabled:
                self._out_audio_source = rtc.AudioSource(
                    self._params.audio_out_sample_rate, self._params.audio_out_channels
                )
                self._out_audio_track = rtc.LocalAudioTrack.create_audio_track(
                    "achatbot-out-audio", self._out_audio_source
                )
                options = rtc.TrackPublishOptions()
                options.source = rtc.TrackSource.SOURCE_MICROPHONE
                publication = await self._room.local_participant.publish_track(
                    self._out_audio_track, options
                )
                logging.info(f"audio track local publication {publication}")

            # Set up video source and track
            # for pub/write out to local_participant camera
            if self._params.camera_out_enabled:
                self._out_video_source = rtc.VideoSource(
                    self._params.camera_out_width,
                    self._params.camera_out_height,
                )
                self._out_video_track = rtc.LocalVideoTrack.create_video_track(
                    "achatbot-video", self._out_video_source
                )
                options = rtc.TrackPublishOptions()
                options.source = rtc.TrackSource.SOURCE_CAMERA
                publication = await self._room.local_participant.publish_track(
                    self._out_video_track, options
                )
                logging.info(f"video track local publication {publication}")

            # Check if there are already participants in the room
            participants = self.get_participants()
            if len(participants) > 0 and not self._other_participant_has_joined:
                logging.info(f"first participant {participants[0]} join")
                self._other_participant_has_joined = True
                # TODO: need check who is 主播,
                # default use the first remote participants[0],
                # or callback params use room to broadcast
                await self._callbacks.on_first_participant_joined(participants[0])

            room_url = f"{self._params.sandbox_room_url}/rooms/{self._room_name}"
            logging.info(f"u can access sandbox url: {room_url}")
        except asyncio.TimeoutError:
            error_msg = f"Time out joining {self._room_name}"
            logging.error(error_msg, exc_info=True)
            await self._callbacks.on_error(error_msg)
        except Exception as e:
            error_msg = f"Error join {self._room_name}: {e}"
            logging.error(error_msg, exc_info=True)
            await self._callbacks.on_error(error_msg)

    async def _leave(self):
        # if end processor, leave room
        # need manual touch disconnect callback,some reason disconnect fail
        await self._async_on_disconnected(reason="Leave Room.")
        self._in_local_audio_stream and await self._in_local_audio_stream.aclose()
        self._participant_audio_stream and await self._participant_audio_stream.aclose()
        self._capture_participant_audio_stream and await (
            self._capture_participant_audio_stream.aclose()
        )
        self._capture_participant_video_stream and await (
            self._capture_participant_video_stream.aclose()
        )
        await self._room.disconnect()

    async def leave(self):
        if not self._joined or self._leaving:
            logging.warning(f"{self._join_name} unconnect {self._room_name}, don't to leave")
            return

        self._joined = False
        self._leaving = True

        logging.info(f"{self._join_name} Disconnecting from {self._room_name}")

        # TODO: transcription_enabled to stop

        # if join and leave quick,and some message is async send message
        # need sleep to disconnect or wait completed event
        await asyncio.sleep(1)

        try:
            await self._leave()
        except asyncio.TimeoutError:
            error_msg = f"Time out leaving {self._room_name}"
            logging.error(error_msg, exc_info=True)
            await self._callbacks.on_error(error_msg)
        except Exception as e:
            self._leaving = False
            error_msg = f"Error leave {self._room_name}: {e}"
            logging.error(error_msg, exc_info=True)
            await self._callbacks.on_error(error_msg)

    async def send_message(self, frame: TransportMessageFrame):
        if not self._joined:
            return

        destination_identities = []
        if isinstance(frame, LivekitTransportMessageFrame):
            if frame.participant_id:
                destination_identities.append(frame.participant_id)
        # TODO: need use pb or msgpkg with fe sdk(metrics) @weedge
        data = pickle.dumps(frame.message)

        try:
            await self._room.local_participant.publish_data(
                data,
                reliable=True,
                destination_identities=destination_identities,
            )
        except Exception as e:
            logging.error(f"Error sending data: {e}")

    def get_participant_ids(self) -> List[str]:
        return [p.sid for p in self._room.remote_participants.values()]

    def get_participants(self) -> List[rtc.RemoteParticipant]:
        #!NOTE: python < 3.6 dict is un order
        return list(self._room.remote_participants.values())

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

    def _on_connection_state_changed_wrapper(self, state: rtc.ConnectionState):
        asyncio.create_task(self._async_on_connection_state_changed(state))

    def _on_disconnected_wrapper(self, reason: protocol.models.DisconnectReason):
        asyncio.create_task(self._async_on_disconnected(str(reason)))

    # Async methods for event handling
    async def _async_on_participant_connected(self, participant: rtc.RemoteParticipant):
        logging.info(f"Participant:{participant} connected")
        await self._callbacks.on_participant_connected(participant)
        if not self._other_participant_has_joined:
            self._other_participant_has_joined = True
            await self._callbacks.on_first_participant_joined(participant)

    async def _async_on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        logging.info(f"Participant:{participant} disconnected")
        await self._callbacks.on_participant_disconnected(participant)
        if len(self.get_participants()) == 0:
            self._other_participant_has_joined = False

    async def _async_on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logging.info(f"track subscribed: {track} from participant {participant}")
        if track.kind == rtc.TrackKind.KIND_AUDIO and (
            self._params.audio_in_enabled or self._params.vad_enabled
        ):
            if self._params.audio_in_participant_enabled:
                # dispatch particpant track for capture participant audio stream
                self._in_participant_audio_tracks[participant.sid] = track
            else:
                # sub a new particpant track stream
                self._participant_audio_stream = rtc.AudioStream(
                    track=track,
                    sample_rate=self._params.audio_in_sample_rate,
                    num_channels=self._params.audio_in_channels,
                )
                self._in_audio_task = asyncio.create_task(
                    self._process_audio_stream(self._participant_audio_stream, participant.sid)
                )
            await self._callbacks.on_audio_track_subscribed(participant)
        elif track.kind == rtc.TrackKind.KIND_VIDEO and self._params.camera_in_enabled:
            self._in_participant_video_tracks[participant.sid] = track
            await self._callbacks.on_video_track_subscribed(participant)

    async def _process_audio_stream(self, audio_stream: rtc.AudioStream, participant_id: str):
        logging.info(f"Started processing audio stream for participant {participant_id}")
        async for event in audio_stream:
            if isinstance(event, rtc.AudioFrameEvent):
                await self._in_audio_queue.put((event, participant_id))

    async def _async_on_track_unsubscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logging.info(f"track unsubscribed: {track} from participant {participant}")
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            if participant.sid in self._in_participant_audio_tracks:
                self._in_participant_audio_tracks.pop(participant.sid)
            await self._callbacks.on_audio_track_unsubscribed(participant)
        elif track.kind == rtc.TrackKind.KIND_VIDEO:
            if participant.sid in self._in_participant_video_tracks:
                self._in_participant_video_tracks.pop(participant.sid)
            await self._callbacks.on_video_track_unsubscribed(participant)

    async def _async_on_data_received(self, data: rtc.DataPacket):
        await self._callbacks.on_data_received(data.data, data.participant)

    async def _async_on_connected(self):
        await self._callbacks.on_connected(self._room)

    async def _async_on_connection_state_changed(self, state: rtc.ConnectionState):
        await self._callbacks.on_connection_state_changed(state)

    async def _async_on_disconnected(self, reason: str):
        self._joined = False
        logging.info(f"Disconnected from {self._room_name}. Reason: {reason}")
        await self._callbacks.on_disconnected(reason)

    # Audio in

    async def read_next_audio_frame(self) -> AudioRawFrame | None:
        """get room sub all participant audio frame from a queue"""
        audio_frame_event, participant_id = await self._in_audio_queue.get()
        audio_frame = self._convert_input_audio(audio_frame_event, participant_id)
        return audio_frame

    def capture_participant_audio(
        self,
        participant_id: str,
        callback: Callable[[UserAudioRawFrame], Awaitable[None]],
        sample_rate=None,
        num_channels=None,
    ):
        # just capture one participant audio
        audio_track = self._in_participant_audio_tracks.get(participant_id)
        if not audio_track:
            logging.warning(f"participant_id {participant_id} no audio track")
            return

        if (
            self._current_sub_audio_participant_id
            and self._capture_participant_audio_stream
            and self._current_sub_audio_participant_id == participant_id
        ):
            logging.info(
                f"participant_id {participant_id} has captured aduio stream: {self._capture_participant_audio_stream}"
            )
            return

        # switch audio stream
        if (
            self._current_sub_audio_participant_id
            and self._capture_participant_audio_stream
            and self._current_sub_audio_participant_id != participant_id
        ):
            self._current_sub_audio_participant_id = participant_id
            asyncio.create_task(self._capture_participant_audio_stream.aclose())

        self._capture_participant_audio_stream = rtc.AudioStream(
            audio_track,
            sample_rate=sample_rate if sample_rate else self._params.audio_in_sample_rate,
            num_channels=num_channels if num_channels else self._params.audio_in_channels,
        )
        self._on_participant_audio_frame_task = asyncio.create_task(
            self._async_on_participant_audio_frame(
                participant_id, callback, self._capture_participant_audio_stream
            )
        )

    async def _async_on_participant_audio_frame(
        self,
        participant_id: str,
        callback: Callable[[UserAudioRawFrame], Awaitable[None]],
        audio_stream: rtc.AudioStream,
    ):
        async for audio_frame_event in audio_stream:
            try:
                if isinstance(audio_frame_event, rtc.AudioFrameEvent):
                    audio_frame = self._convert_input_audio(
                        audio_frame_event,
                        participant_id,
                    )
                    if asyncio.iscoroutinefunction(callback):
                        await callback(audio_frame)
                    else:
                        callback(audio_frame)
                else:
                    logging.warning(
                        f"Received unexpected event type: {type(audio_frame_event)} and participant {participant_id}"
                    )
            except asyncio.CancelledError:
                await audio_stream.aclose()
                logging.info("task cancelled")
                break
            except Exception as e:
                logging.error(f"task Error: {e}", exc_info=True)

    async def read_participant_next_audio_frame_iter(
        self, participant_id: str
    ) -> AsyncGenerator[AudioRawFrame, None]:
        """read next audio frame from remote participant audio track stream iter"""
        audio_track = self._in_participant_audio_tracks.get(participant_id)
        if not audio_track:
            yield None
        audio_stream = rtc.AudioStream(
            audio_track,
            sample_rate=self._params.audio_in_sample_rate,
            num_channels=self._params.audio_in_channels,
        )
        async for audio_frame_event in audio_stream:
            audio_frame = self._convert_input_audio(audio_frame_event, participant_id)
            yield audio_frame
        await audio_stream.aclose()

    async def read_next_audio_frame_iter(self) -> AsyncGenerator[AudioRawFrame, None]:
        """read next audio frame from local microphone source"""
        if not self._in_local_audio_stream:
            yield None
        async for audio_frame_event in self._in_local_audio_stream:
            audio_frame = self._convert_input_audio(
                audio_frame_event,
                self._room.local_participant.sid,
            )
            yield audio_frame

    def _convert_input_audio(
        self,
        audio_frame_event: rtc.AudioFrameEvent,
        participant_id: str = "",
    ) -> UserAudioRawFrame | None:
        audio_data = audio_frame_event.frame.data
        pcm_data = np.frombuffer(audio_data, dtype=np.int16)
        original_sample_rate = audio_frame_event.frame.sample_rate
        original_num_channels = audio_frame_event.frame.num_channels

        # Allow 8kHz and 16kHz, other sampple rate convert to 16kHz
        if original_sample_rate not in [8000, 16000]:
            sample_rate = 16000
            pcm_data = resample_audio(pcm_data, original_sample_rate, sample_rate)
        else:
            sample_rate = original_sample_rate

        if sample_rate != self._params.audio_in_sample_rate:
            self._params.audio_in_sample_rate = sample_rate
            if self._params.vad_enabled and self._params.vad_analyzer:
                self._params.vad_analyzer.set_args(
                    {
                        "sample_rate": sample_rate,
                        "num_channels": original_num_channels,
                    }
                )

        return UserAudioRawFrame(
            user_id=participant_id,
            audio=pcm_data.tobytes(),
            sample_rate=sample_rate,
            num_channels=original_num_channels,
        )

    # Audio out

    async def write_raw_audio_frames(
        self,
        frames: bytes,
    ):
        if not self._joined or not self._out_audio_source:
            return

        try:
            audio_frame: rtc.AudioFrame = self._convert_output_audio(frames)
            await self._out_audio_source.capture_frame(audio_frame)
        except Exception as e:
            logging.error(f"Error publishing audio: {e}", exc_info=True)

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

    def _convert_input_video_image(
        self,
        participant_id: str,
        video_frame_event: rtc.VideoFrameEvent,
        target_color_mode: str = "RGB",
    ) -> UserImageRawFrame | None:
        buffer = video_frame_event.frame

        match buffer.type:
            case rtc.VideoBufferType.RGBA:
                image = Image.frombuffer(
                    "RGBA", (buffer.width, buffer.height), buffer.data, "raw", "RGBA", 0, 1
                )
            case rtc.VideoBufferType.ABGR:
                image = Image.frombuffer(
                    "ABGR", (buffer.width, buffer.height), buffer.data, "raw", "ABGR", 0, 1
                )
            case rtc.VideoBufferType.ARGB:
                image = Image.frombuffer(
                    "ARGB", (buffer.width, buffer.height), buffer.data, "raw", "ARGB", 0, 1
                )
            case rtc.VideoBufferType.BGRA:
                image = Image.frombuffer(
                    "BGRA", (buffer.width, buffer.height), buffer.data, "raw", "BGRA", 0, 1
                )
            case rtc.VideoBufferType.RGB24:
                image = Image.frombuffer(
                    "RGB", (buffer.width, buffer.height), buffer.data, "raw", "RGB", 0, 1
                )
            case _:
                logging.warning(f"buffer type:{buffer.type} un support convert")
                return None

        image = image.convert(target_color_mode)
        return UserImageRawFrame(
            user_id=participant_id,
            image=image.tobytes(),
            size=(buffer.width, buffer.height),
            mode=target_color_mode,
            format="JPEG",
        )

    def _get_livekit_video_buffer_type(self):
        livekit_video_buffer_type = rtc.VideoBufferType.RGB24
        match self._params.camera_in_color_format:
            case "RGB":
                livekit_video_buffer_type = rtc.VideoBufferType.RGB24  # jpeg
            case "RGBA":
                livekit_video_buffer_type = rtc.VideoBufferType.RGBA  # png
            case "_":
                livekit_video_buffer_type = rtc.VideoBufferType.RGB24  # other defualt RGB24
        return livekit_video_buffer_type

    def capture_participant_video(
        self,
        participant_id: str,
        callback: Callable[[UserImageRawFrame], Awaitable[None]],
        framerate: int = 30,
        video_source: str = "camera",
        color_format: str = "RGB",
    ):
        # just capture one participant video
        video_track = self._in_participant_video_tracks.get(participant_id)
        if not video_track:
            logging.warning(f"participant_id {participant_id} no video track")
            return

        if (
            self._current_sub_video_participant_id
            and self._capture_participant_video_stream
            and self._current_sub_video_participant_id == participant_id
        ):
            logging.info(
                f"participant_id {participant_id} has captured video stream: {self._capture_participant_video_stream}"
            )
            return

        # switch video stream
        if (
            self._current_sub_video_participant_id
            and self._capture_participant_video_stream
            and self._current_sub_video_participant_id != participant_id
        ):
            self._current_sub_video_participant_id = participant_id
            asyncio.create_task(self._capture_participant_video_stream.aclose())

        self._capture_participant_video_stream = rtc.VideoStream(
            video_track,
            format=self._get_livekit_video_buffer_type(),
        )

        self._on_participant_video_frame_task = asyncio.create_task(
            self._async_on_participant_video_frame(
                participant_id, callback, self._capture_participant_video_stream, color_format
            )
        )

    async def _async_on_participant_video_frame(
        self,
        participant_id: str,
        callback: Callable[[UserImageRawFrame], Awaitable[None]],
        video_stream: rtc.VideoStream,
        color_format: str = "RGB",
    ):
        logging.info(
            f"Started capture participant_id:{participant_id} from video stream {video_stream}"
        )
        async for video_frame_event in video_stream:
            try:
                if isinstance(video_frame_event, rtc.VideoFrameEvent):
                    image_frame = self._convert_input_video_image(
                        participant_id,
                        video_frame_event,
                        target_color_mode=color_format,
                    )
                    if asyncio.iscoroutinefunction(callback):
                        await callback(image_frame)
                    else:
                        callback(image_frame)
                else:
                    logging.warning(
                        f"Received unexpected event type: {type(video_frame_event)} and participant {participant_id}"
                    )
            except asyncio.CancelledError:
                await video_stream.aclose()
                logging.info("task cancelled")
                break
            except Exception as e:
                logging.error(f"task Error: {e}", exc_info=True)

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
            frame.image,
        )
        self._out_video_source and self._out_video_source.capture_frame(video_frame)
