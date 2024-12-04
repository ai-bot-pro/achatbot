import os
import json
import logging
import asyncio
from urllib.parse import quote
from typing import Any, Awaitable, Callable, List

import numpy as np
from PIL import Image
from pydantic import BaseModel
from apipeline.frames.data_frames import AudioRawFrame, ImageRawFrame

from src.services.help.agora.token import TokenClaims, TokenPaser
from src.common.types import SAMPLE_WIDTH, AgoraParams, LOG_DIR
from src.common.utils.audio_utils import resample_audio
from src.types.frames.data_frames import AgoraTransportMessageFrame, TransportMessageFrame, UserAudioRawFrame, UserImageRawFrame

try:
    from agora_realtime_ai_api import rtc
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use Agora, you need to `pip install achatbot[agora]`.")
    raise Exception(f"Missing module: {e}")


class AgoraService:
    @staticmethod
    def init(app_id: str) -> rtc.AgoraService:
        config = rtc.AgoraServiceConfig()
        config.audio_scenario = rtc.AudioScenarioType.AUDIO_SCENARIO_CHORUS
        config.appid = app_id
        config.log_path = os.path.join(LOG_DIR, "agorasdk.log")

        agora_service = rtc.AgoraService()
        agora_service.initialize(config)

        return agora_service


class RtcChannelEventObserver(rtc.ChannelEventObserver):
    """
    if rtc.ChannelEventObserver no register event name,
    - need add for custom event to run
    - or override event name to run
    """

    def on_connection_failure(
        self, agora_rtc_conn: rtc.RTCConnection, conn_info: rtc.RTCConnInfo, reason: int
    ):
        super().on_connection_failure(agora_rtc_conn, conn_info, reason)
        # logging.error(f"Connected to RTC: {agora_rtc_conn} {conn_info} {reason}")
        self.emit_event("connection_failure", agora_rtc_conn, conn_info, reason)

    def on_connected(
        self, agora_rtc_conn: rtc.RTCConnection, conn_info: rtc.RTCConnInfo, reason: int
    ):
        super().on_connected(agora_rtc_conn, conn_info, reason)
        # logging.info(f"Connected to RTC: {agora_rtc_conn} {conn_info} {reason}")
        self.emit_event("connected", agora_rtc_conn, conn_info, reason)

    def on_disconnected(
        self, agora_rtc_conn: rtc.RTCConnection, conn_info: rtc.RTCConnInfo, reason: int
    ):
        super().on_disconnected(agora_rtc_conn, conn_info, reason)
        # logging.info(f"Disconnected to RTC: {agora_rtc_conn} {conn_info} {reason}")
        self.emit_event("disconnected", agora_rtc_conn, conn_info, reason)

    def on_playback_audio_frame_before_mixing(
        self, agora_local_user: rtc.LocalUser, channelId, uid, frame: rtc.AudioFrame
    ):
        logging.debug(f"push AudioStream from rtc.AudioFrame:{frame}")
        return super().on_playback_audio_frame_before_mixing(
            agora_local_user, channelId, uid, frame)


class RtcChannel(rtc.Channel):
    """
    rtc channel now just support audio now
    TODO: support video from https://github.com/AgoraIO-Extensions/Agora-Python-Server-SDK
    """

    def __init__(
        self,
        token_claims: TokenClaims,
        options: rtc.RtcOptions,
        loop: asyncio.AbstractEventLoop,
        service: rtc.AgoraService = None,
    ) -> None:
        self._token_claims = token_claims
        self.token = token_claims.token

        self.options = options
        self.uid = options.uid
        self.channelId = options.channel_name
        self.enable_pcm_dump = options.enable_pcm_dump

        self.loop = loop
        self.stream_message_queue = asyncio.Queue()

        self._service = service if service else AgoraService.init(token_claims.app_id)

        # chat message
        self.chat = rtc.Chat(self)

        # connect
        self.connection_state = 0
        self.remote_users = dict[int, Any]()
        conn_config = rtc.RTCConnConfig(
            client_role_type=rtc.ClientRoleType.CLIENT_ROLE_BROADCASTER,
            channel_profile=rtc.ChannelProfileType.CHANNEL_PROFILE_LIVE_BROADCASTING,
        )
        self.connection = self._service.create_rtc_connection(conn_config)

        # Create the event emitter
        self.emitter = rtc.AsyncIOEventEmitter(loop)
        # Channel Event Observer, register event name
        self.channel_event_observer = RtcChannelEventObserver(
            self.emitter,
            options=options,
        )
        self.connection.register_observer(self.channel_event_observer)

        # local user
        self.local_user = self.connection.get_local_user()
        self.local_user.set_playback_audio_frame_before_mixing_parameters(
            options.channels, options.sample_rate
        )
        self.local_user.register_local_user_observer(self.channel_event_observer)
        self.local_user.register_audio_frame_observer(self.channel_event_observer)
        # self.local_user.subscribe_all_audio()

        # audio media node factory for out audio track
        self.media_node_factory = self._service.create_media_node_factory()
        self.audio_pcm_data_sender = (
            self.media_node_factory.create_audio_pcm_data_sender()
        )
        self.audio_track = self._service.create_custom_audio_track_pcm(
            self.audio_pcm_data_sender
        )

        # create a data stream
        self.stream_id = self.connection.create_data_stream(False, False)

        # create a task to process stream message
        def log_exception(t: asyncio.Task[Any]) -> None:
            if not t.cancelled() and t.exception():
                logging.error("unhandled exception", exc_info=t.exception())
        asyncio.create_task(self._process_stream_message()).add_done_callback(log_exception)

        # emitter event callbacks (event can register multiple callbacks)
        # NOTE:register emit event name see rtc.ChannelEventObserver,
        # c/c++ rtc core lib(defined callback impl) to call and register callback
        self.on("user_joined", self.on_user_joined)
        self.on("connection_state_changed", self.on_connection_state_changed)
        self.on("audio_subscribe_state_changed", self.on_audio_subscribe_state_changed)
        self.on("user_left", self.on_user_left)

    def set_local_user_audio_track(self) -> None:
        self.audio_track.set_enabled(1)
        self.local_user.publish_audio(self.audio_track)

    def set_local_user_video_track(self) -> None:
        """TODO:  support publish video track"""
        pass

    # Event Callback

    def on_connection_state_changed(self, agora_rtc_conn, conn_info, reason):
        # logging.debug(f"conn state {conn_info.state}")
        self.connection_state = conn_info.state

    def on_user_joined(self, agora_rtc_conn, user_id):
        # logging.debug(f"User {user_id} joined")
        self.remote_users[user_id] = True

    def on_user_left(self, agora_rtc_conn, user_id, reason):
        # logging.debug(f"User {user_id} left")
        if user_id in self.remote_users:
            self.remote_users.pop(user_id, None)
        if user_id in self.channel_event_observer.audio_streams:
            audio_stream = self.channel_event_observer.audio_streams.pop(user_id, None)
            audio_stream.queue.put_nowait(None)

    def on_audio_subscribe_state_changed(
        self,
        agora_local_user: rtc.LocalUser,
        channel: str,
        user_id: int,
        old_state: int,
        new_state: int,
        elapse_since_last_state: int,
    ):
        # logging.debug(f"{agora_local_user} {channel} {user_id} {old_state} {new_state}")
        if new_state == 3:  # Successfully subscribed
            if user_id not in self.channel_event_observer.audio_streams:
                self.channel_event_observer.audio_streams[user_id] = rtc.AudioStream()


class AgoraCallbacks(BaseModel):
    """async callback"""
    on_connected: Callable[[rtc.RTCConnection, rtc.RTCConnInfo, int], Awaitable[None]]
    on_connection_state_changed: Callable[[
        rtc.RTCConnection, rtc.RTCConnInfo, int], Awaitable[None]]
    on_connection_failure: Callable[[int], Awaitable[None]]
    on_disconnected: Callable[[rtc.RTCConnection, rtc.RTCConnInfo, int], Awaitable[None]]
    on_error: Callable[[str], Awaitable[None]]
    on_participant_connected: Callable[[rtc.RTCConnection, int], Awaitable[None]]
    on_participant_disconnected: Callable[[rtc.RTCConnection, int, int], Awaitable[None]]
    on_data_received: Callable[[bytes, int], Awaitable[None]]
    on_first_participant_joined: Callable[[rtc.RTCConnection, int], Awaitable[None]]
    on_audio_subscribe_state_changed: Callable[[
        rtc.LocalUser, str, int, int, int, int], Awaitable[None]]


class AgoraTransportClient:
    def __init__(
        self,
        token: str,
        params: AgoraParams,
        callbacks: AgoraCallbacks,
        loop: asyncio.AbstractEventLoop,
        service: rtc.AgoraService = None,
    ):
        self._token = token
        self._token_claims: TokenClaims = None
        if not self.verify_token():
            raise Exception(f"token {token} is not valid")

        self._app_id = self._token_claims.app_id
        self._room_name = self._token_claims.rtc.channel_name
        self._join_name = self._token_claims.rtc.uid
        self._params = params
        self._callbacks = callbacks
        self._loop = loop
        self._service = service if service else AgoraService.init(self._app_id)
        self._channel = RtcChannel(self._token_claims, rtc.RtcOptions(
            channel_name=self._token_claims.rtc.channel_name,
            uid=self._token_claims.rtc.uid,
            sample_rate=params.audio_in_sample_rate,
            channels=params.audio_in_channels,
            enable_pcm_dump=params.enable_pcm_dump,
        ), self._loop, self._service)

        self._joined = False
        self._joining = False
        self._leaving = False

        self._other_participant_has_joined = False

        # audio out
        # local participant audio stream

        # audio in
        # TODO: switch room anchor participant, need know room anchor
        self._in_audio_queue = asyncio.Queue[bytes]()
        self._in_audio_task: asyncio.Task = self._loop.create_task(self.in_audio_handle())

        # Set up channel event handlers to call register handlers
        # NOTE: those event at last will be called for customized callback
        self._channel.on("user_joined", self.on_participant_connected)
        self._channel.on("user_left", self.on_participant_disconnected)
        self._channel.on("stream_message", self.on_data_received)
        self._channel.on("connected", self.on_connected)
        self._channel.on("disconnected", self.on_disconnected)
        self._channel.on("connection_state_changed", self.on_connection_state_changed)
        self._channel.on("connection_failure", self.on_connection_failure)
        self._channel.on("audio_subscribe_state_changed", self.on_audio_subscribe_state_changed)

    def verify_token(self) -> bool:
        try:
            self._token_claims = TokenPaser.parse_claims(self._token)
        except Exception as e:
            logging.warning(f"verfiy {self._token} Exception: {e}")
            return False

        return True

    async def cleanup(self):
        if self._in_audio_task\
                and not self._in_audio_task.cancelled():
            self._in_audio_task.cancel()
            await self._in_audio_task
            # logging.info("Cancelled in_audio_task")

    async def _join(self):
        self._joining = True
        participants = self.get_participant_ids()
        logging.info(f"{self._join_name} Connecting to {self._room_name},"
                     f" current remote participants:{participants}")

        try:
            await self._channel.connect()
        except Exception as e:
            self._joining = False
            self._joined = False
            raise e

        logging.info(
            f"local_participant:{self._channel.uid} joined room(channel): {self._room_name} connection_state: {self._channel.connection_state}")
        self._joining = False
        self._joined = True

    async def join(self):
        # input and output procssor maybe do join room at the same time
        # need check join state, avoid duplicate join
        if self._joined or self._joining:
            logging.warning(f"{self._join_name} has connected {self._room_name}")
            return

        try:
            await self._join()

            # TODO: transcription_enabled to start

            demo_url = self._params.demo_voice_url
            # Set up audio source and track
            # - for pub/write out to local_participant microphone
            if self._params.audio_out_enabled:
                self._channel.set_local_user_audio_track()
                logging.info(f"local user audio track enabled")

            # Set up video source and track
            # for pub/write out to local_participant camera
            if self._params.camera_out_enabled:
                self._channel.set_local_user_video_track()
                logging.info(f"local user video track enabled")
                demo_url = self._params.demo_video_url

            # Check if there are already participants in the room
            participants = self.get_participant_ids()
            if len(participants) > 0 and not self._other_participant_has_joined:
                logging.info(f"first participant {participants[0]} join")
                self._other_participant_has_joined = True
                # TODO: need check who is 主播,
                # default use the first remote participants[0],
                # or callback params use room to broadcast

                await self._callbacks.on_first_participant_joined(participants[0])

            url = f"{demo_url}?appid={quote(self._app_id)}&channel={quote(self._room_name)}&token={quote(self._token)}&uid={quote(str(self._token_claims.rtc.uid))}"
            logging.info(f"u can access url: {url}")
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
        # await self._async_on_disconnected(reason="Leave Room.")
        await self._channel.disconnect()

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
        # await asyncio.sleep(1)

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

    async def wait_for_remote_user(self, timeout_s: float | None = None) -> int:
        """
        if have remote users joined, return first user id;
        else wait for remote user join. (wait with timeout_s, timeout_s is None , no wait timeout)
        return user id
        """
        remote_users = self.get_participant_ids()
        if len(remote_users) > 0:
            return remote_users[0]

        future = asyncio.Future[int]()

        # do user joined callback once
        self._channel.once("user_joined", lambda conn, user_id: future.set_result(user_id))

        try:
            # Wait for the remote user with a timeout_s, timeout_s is None , no wait timeout
            remote_user = await asyncio.wait_for(future, timeout=timeout_s)
            logging.info(f"Subscribing from remote_user {remote_user}")
            return remote_user
        except asyncio.TimeoutError:
            future.cancel()
        except KeyboardInterrupt:
            future.cancel()
        except Exception as e:
            logging.error(f"Error waiting for remote user: {e}")
            raise

    async def send_message(self, frame: TransportMessageFrame):
        """
        Send a chat message in the chat 1v1 channel.
        TODO: send signaling message to remote_participant with rtm
        """
        if not self._joined:
            return

        try:
            await self._channel.chat.send_message(
                rtc.ChatMessage(
                    message=json.dumps(frame.message),
                    # need use message id
                    msg_id=frame.participant_id,
                )
            )
        except Exception as e:
            logging.error(f"Error sending data: {e}")

    def get_participant_ids(self) -> List[int]:
        return list(self._channel.remote_users.keys())

    async def get_participant_metadata(self, participant_id: str) -> dict:
        participant = self._channel.remote_users.get(participant_id)
        if participant:
            # TODO: get remote_participant_info
            return {}
        return {}

    # Event Callback

    def on_participant_connected(
            self, agora_rtc_conn: rtc.RTCConnection, user_id: int):
        if not self._other_participant_has_joined:
            self._other_participant_has_joined = True
            # subscribe audio from the first participant
            self._channel.local_user.subscribe_audio(user_id)
            asyncio.create_task(self._callbacks.on_first_participant_joined(
                agora_rtc_conn, user_id))

        asyncio.create_task(self._callbacks.on_participant_connected(
            agora_rtc_conn, user_id))

    def on_participant_disconnected(
            self,
            agora_rtc_conn: rtc.RTCConnection,
            user_id: int, reason: int):
        if len(self.get_participant_ids()) == 0:
            self._other_participant_has_joined = False
        asyncio.create_task(self._callbacks.on_participant_disconnected(
            agora_rtc_conn, user_id, reason))

    def on_data_received(
            self,
            agora_local_user: rtc.LocalUser,
            user_id: int, stream_id: str, data: bytes, length: int):
        logging.info(
            f"{agora_local_user} Received stream({stream_id}) message from {user_id} with ({type(data)})length: {length}")
        asyncio.create_task(self._callbacks.on_data_received(
            data, user_id))

    def on_connected(
            self, agora_rtc_conn: rtc.RTCConnection, conn_info: rtc.RTCConnInfo, reason: int):
        asyncio.create_task(self._callbacks.on_connected(
            agora_rtc_conn, conn_info, reason))

    def on_connection_state_changed(
            self, agora_rtc_conn: rtc.RTCConnection, conn_info: rtc.RTCConnInfo, reason: int):
        asyncio.create_task(self._callbacks.on_connection_state_changed(
            agora_rtc_conn, conn_info, reason))

    def on_disconnected(
            self, agora_rtc_conn: rtc.RTCConnection, conn_info: rtc.RTCConnInfo, reason: int):
        self._joined = False
        asyncio.create_task(self._callbacks.on_disconnected(
            agora_rtc_conn, conn_info, reason))

    def on_connection_failure(
            self, agora_rtc_conn: rtc.RTCConnection, conn_info: rtc.RTCConnInfo, reason: int):
        self._joined = False
        asyncio.create_task(self._callbacks.on_connection_failure(
            reason))

    def on_audio_subscribe_state_changed(
        self,
        agora_local_user: rtc.LocalUser,
        channel: str,
        user_id: int,
        old_state: int,
        new_state: int,
        elapse_since_last_state: int,
    ):
        asyncio.create_task(self._callbacks.on_audio_subscribe_state_changed(
            agora_local_user, channel, user_id, old_state, new_state, elapse_since_last_state))

    # Audio in

    async def in_audio_handle(self) -> None:
        """sub the first participant audio frame """
        logging.debug("start in_audio_handle")
        try:
            subscribe_user = await self.wait_for_remote_user()
            logging.info(f"subscribe user {subscribe_user}")
            await self._channel.subscribe_audio(subscribe_user)

            # from on_playback_audio_frame_before_mixing callback
            # to get the audio frame published by the remote user
            while subscribe_user is None or self._channel.get_audio_frames(subscribe_user) is None:
                await asyncio.sleep(0.1)

            audio_frames = self._channel.get_audio_frames(subscribe_user)
            async for audio_frame in audio_frames:
                await self._in_audio_queue.put((audio_frame, subscribe_user))
                # Yield control to allow other tasks to run
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            logging.info("Cancelled Audio task")
            return

    async def read_next_audio_frame(self) -> AudioRawFrame | None:
        """get room sub the first participant audio frame from a queue"""
        pcm_audio_frame, participant_id = await self._in_audio_queue.get()
        audio_frame = self._convert_input_audio(pcm_audio_frame, participant_id)
        return audio_frame

    def _convert_input_audio(
        self,
        pcm_audio_frame: rtc.PcmAudioFrame,
        participant_id: str = "",
    ) -> UserAudioRawFrame | None:
        audio_data = pcm_audio_frame.data
        pcm_data = np.frombuffer(audio_data, dtype=np.int16)
        original_sample_rate = pcm_audio_frame.sample_rate
        original_num_channels = pcm_audio_frame.number_of_channels

        # Allow 8kHz and 16kHz, other sampple rate convert to 16kHz
        if original_sample_rate not in [8000, 16000]:
            sample_rate = 16000
            pcm_data = resample_audio(
                pcm_data, original_sample_rate, sample_rate)
        else:
            sample_rate = original_sample_rate

        if sample_rate != self._params.audio_in_sample_rate:
            self._params.audio_in_sample_rate = sample_rate
            if self._params.vad_enabled and self._params.vad_analyzer:
                self._params.vad_analyzer.set_args({
                    "sample_rate": sample_rate,
                    "num_channels": original_num_channels,
                })

        return UserAudioRawFrame(
            user_id=participant_id,
            audio=pcm_data.tobytes(),
            sample_rate=sample_rate,
            num_channels=original_num_channels,
        )

    def capture_participant_audio(
        self,
        participant_id: str,
        callback: Callable[[UserAudioRawFrame], Awaitable[None]],
        sample_rate=None,
        num_channels=None,
    ):
        # TODO:
        pass

    # Audio out

    async def write_raw_audio_frames(self, frame_data: bytes,):
        if not self._joined:
            return

        try:
            # audio_frame: rtc.PcmAudioFrame = self._convert_output_audio(frames)
            await self._channel.push_audio_frame(frame_data)
        except Exception as e:
            logging.error(f"Error publishing audio: {e}", exc_info=True)

    def _convert_output_audio(self, audio_data: bytes) -> rtc.PcmAudioFrame:
        bytes_per_sample = SAMPLE_WIDTH  # Assuming 16-bit audio sample_width
        total_samples = len(audio_data) // bytes_per_sample
        samples_per_channel = total_samples // self._params.audio_out_channels

        frame = rtc.PcmAudioFrame()
        frame.data = audio_data
        frame.bytes_per_sample = bytes_per_sample
        frame.samples_per_channel = samples_per_channel
        frame.sample_rate = self._params.audio_out_sample_rate
        frame.number_of_channels = self._params.audio_out_channels

        return frame

    # Camera in

    def capture_participant_video(
        self,
        participant_id: str,
        callback: Callable[[UserImageRawFrame], Awaitable[None]],
        framerate: int = 30,
        video_source: str = "camera",
        color_format: str = "RGB"
    ):
        # just capture one participant video
        # TODO
        pass

    # Camera out

    async def write_frame_to_camera(self, frame: ImageRawFrame):
        """
        publish image frame to camera
        """
        # TODO
        pass
