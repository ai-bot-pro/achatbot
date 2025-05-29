import copy
import os
import json
import logging
import asyncio
import time
from urllib.parse import quote
from typing import Any, AsyncIterator, Awaitable, Callable, List
import uuid

import PIL.Image
import numpy as np
from pydantic import BaseModel
from apipeline.frames.data_frames import AudioRawFrame, ImageRawFrame

from src.common import const
from src.common.types import SAMPLE_WIDTH, AgoraParams, LOG_DIR, VIDEOS_DIR
from src.common.utils.audio_utils import resample_audio
from src.types.frames.data_frames import (
    AgoraTransportMessageFrame,
    TransportMessageFrame,
    UserAudioRawFrame,
    UserImageRawFrame,
)
from src.services.help.agora.token import TokenClaims, TokenPaser
from src.services.help.agora.video_utils import *

try:
    from agora_realtime_ai_api import rtc
    from agora.rtc.voice_detection import AudioVadV2, AudioVadConfigV2
    from agora.rtc.video_frame_observer import VideoFrame
    from agora.rtc.video_frame_observer import IVideoFrameObserver
    from agora.rtc.video_frame_sender import ExternalVideoFrame
    from agora.rtc.agora_base import (
        VideoEncoderConfiguration,
        VideoDimensions,
        VideoSubscriptionOptions,
    )
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error("In order to use Agora, you need to `pip install achatbot[agora]`.")
    raise Exception(f"Missing module: {e}")


class AgoraService:
    @staticmethod
    def init(params: AgoraParams) -> rtc.AgoraService:
        config = rtc.AgoraServiceConfig()
        # https://api-ref.agora.io/en/voice-sdk/react-native/4.x/API/enum_audioscenariotype.html
        config.audio_scenario = rtc.AudioScenarioType.AUDIO_SCENARIO_CHORUS  # default
        # if params.camera_in_enabled:
        #    config.audio_scenario = rtc.AudioScenarioType.AUDIO_SCENARIO_CHATROOM
        config.appid = params.app_id
        config.enable_video = int(params.camera_in_enabled)
        config.log_path = os.path.join(LOG_DIR, "agorasdk.log")

        agora_service = rtc.AgoraService()
        agora_service.initialize(config)

        return agora_service


class VideoStream:
    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue()

    def __aiter__(self) -> AsyncIterator[VideoFrame]:
        return self

    async def __anext__(self) -> VideoFrame:
        item = await self.queue.get()
        if item is None:
            raise StopAsyncIteration

        return item


class RtcChannelEventObserver(IVideoFrameObserver, rtc.ChannelEventObserver):
    """
    if rtc.ChannelEventObserver no register event name,
    - need add for custom event to run
    - or override event name to run

    add IVideoFrameObserver to support sub video frame
    """

    def __init__(self, event_emitter: rtc.AsyncIOEventEmitter, options: rtc.RtcOptions) -> None:
        super().__init__(event_emitter, options)
        self.video_streams = dict[str, VideoStream]()

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
        self,
        agora_local_user: rtc.LocalUser,
        channelId,
        uid,
        frame: rtc.AudioFrame,
        vad_result_state: int,
        vad_result_bytearray: bytearray,  # TODO: vad
    ):
        logging.debug(f"push AudioStream from channel:{channelId} uid:{uid} rtc.AudioFrame:{frame}")

        # fix: when use had joined, the audio_streams no uid audio stream
        if self.audio_streams.get(uid) is None:
            return 0

        return super().on_playback_audio_frame_before_mixing(
            agora_local_user, channelId, uid, frame
        )

    def on_video_subscribe_state_changed(
        self, agora_local_user, channel, user_id, old_state, new_state, elapse_since_last_state
    ):
        self.emit_event(
            "video_subscribe_state_changed",
            agora_local_user,
            channel,
            user_id,
            old_state,
            new_state,
            elapse_since_last_state,
        )

    def on_frame(self, channel_id, remote_uid, video_frame: VideoFrame):
        # on_video_frame, channel_id=room-bot, remote_uid=1867636435, type=1, width=640,
        # height=480, y_stride=640, u_stride=320, v_stride=320, len_y=307200,
        # len_u=76800, len_v=76800, len_alpha_buffer=0
        if remote_uid == "0":  # TODO: check
            logging.debug(
                f"on_video_frame, channel_id={channel_id},"
                f"remote_uid={remote_uid}, type={video_frame.type}, width={video_frame.width},"
                f"height={video_frame.height}, y_stride={video_frame.y_stride},"
                f"u_stride={video_frame.u_stride}, v_stride={video_frame.v_stride},"
                f"len_y={len(video_frame.y_buffer)}, len_u={len(video_frame.u_buffer)},"
                f"len_v={len(video_frame.v_buffer)},"
                f"len_alpha_buffer={len(video_frame.alpha_buffer) if video_frame.alpha_buffer else 0}"
            )
            return 1

        # fix: when use had joined, the video_streams no uid video stream
        if self.video_streams.get(remote_uid) is None:
            return 1

        self.loop.call_soon_threadsafe(self.video_streams[remote_uid].queue.put_nowait, video_frame)
        return 1


class RtcChannel(rtc.Channel):
    """
    rtc channel now support audio/video now
    """

    def __init__(
        self,
        params: AgoraParams,
        token_claims: TokenClaims,
        options: rtc.RtcOptions,
        loop: asyncio.AbstractEventLoop,
        service: rtc.AgoraService = None,
        vad_conf: AudioVadConfigV2 = None,  # TODO:vad
    ) -> None:
        self._param = params
        self._token_claims = token_claims
        self.token = token_claims.token

        self.options = options
        self.uid = options.uid
        self.channelId = options.channel_name
        self.enable_pcm_dump = options.enable_pcm_dump

        self.loop = loop

        self._service = service if service else AgoraService.init(self._param)

        # chat message
        self.chat = rtc.Chat(self)

        # connect
        self.connection_state = 0
        self.remote_users = dict[int, Any]()
        conn_config = rtc.RTCConnConfig(
            client_role_type=rtc.ClientRoleType.CLIENT_ROLE_BROADCASTER,
            channel_profile=rtc.ChannelProfileType.CHANNEL_PROFILE_LIVE_BROADCASTING,
        )
        # NOTE: use subscribe_audio don't to set auto_subscribe_audio
        # if params.audio_in_enabled:
        #    conn_config.auto_subscribe_audio = 1
        # NOTE: use subscribe_video don't to set auto_subscribe_video
        # if params.camera_in_enabled:
        #     conn_config.auto_subscribe_video = 1
        self.connection = self._service.create_rtc_connection(conn_config)

        # Create the event emitter for observer
        self.emitter = rtc.AsyncIOEventEmitter(loop)
        # Channel Event Observer, register event name
        self.channel_event_observer = RtcChannelEventObserver(
            self.emitter,
            options=options,
        )

        # connect regist event observer
        self.connection.register_observer(self.channel_event_observer)
        # local user
        self.local_user = self.connection.get_local_user()

        # register local user event observer
        self.local_user.register_local_user_observer(self.channel_event_observer)
        # 1.1 in audio for local user
        if params.audio_in_enabled:
            self.local_user.set_playback_audio_frame_before_mixing_parameters(
                options.channels, options.sample_rate
            )
            # enable_vad = int(params.vad_enabled)  # TODO: vad
            enable_vad = 0
            ret = self.local_user.register_audio_frame_observer(
                self.channel_event_observer, enable_vad, vad_conf
            )

            if ret < 0:
                raise Exception("register_audio_frame_observer failed")
            # self.local_user.subscribe_all_audio()

        # 1.2 in video for local user
        if params.camera_in_enabled:
            ret = self.local_user.register_video_frame_observer(self.channel_event_observer)
            if ret < 0:
                raise Exception("register_video_frame_observer failed")

        # 2. media node factory for out audio/video track
        self.media_node_factory = self._service.create_media_node_factory()
        # 2.1 out audio track
        if params.audio_out_enabled:
            self.audio_pcm_data_sender = self.media_node_factory.create_audio_pcm_data_sender()
            self.audio_track = self._service.create_custom_audio_track_pcm(
                self.audio_pcm_data_sender
            )

        # 2.2 out video track
        if params.camera_out_enabled:
            self.video_yuv_data_sender = self.media_node_factory.create_video_frame_sender()
            self.video_track = self._service.create_custom_video_track_frame(
                self.video_yuv_data_sender
            )
            video_config = VideoEncoderConfiguration(
                frame_rate=params.camera_out_framerate,
                dimensions=VideoDimensions(
                    width=params.camera_out_width, height=params.camera_out_height
                ),
                encode_alpha=0,
            )
            self.video_track.set_video_encoder_configuration(video_config)

        # 3. create a data stream for send stream message
        self.stream_id = self.connection.create_data_stream(False, False)
        self.stream_message_queue = asyncio.Queue()

        # create a task to process stream message
        def log_exception(t: asyncio.Task[Any]) -> None:
            if not t.cancelled() and t.exception():
                logging.error("unhandled exception", exc_info=t.exception())

        asyncio.create_task(self._process_stream_message()).add_done_callback(log_exception)

        # 4. emitter event callbacks (event can register multiple callbacks)
        # NOTE:register emit event name see rtc.ChannelEventObserver,
        # c/c++ rtc core lib(defined callback impl) to call and register callback
        self.on("user_joined", self.on_user_joined)
        self.on("connection_state_changed", self.on_connection_state_changed)
        self.on("audio_subscribe_state_changed", self.on_audio_subscribe_state_changed)
        self.on("video_subscribe_state_changed", self.on_video_subscribe_state_changed)
        self.on("user_left", self.on_user_left)

    def set_local_user_audio_track(self) -> None:
        if self._param.audio_out_enabled:
            self.audio_track.set_enabled(1)
            self.local_user.publish_audio(self.audio_track)

    def set_local_user_video_track(self) -> None:
        if self._param.camera_out_enabled:
            self.video_track.set_enabled(1)
            self.local_user.publish_video(self.video_track)

    async def clear_sender_video_buffer(self) -> None:
        """
        Clears the video buffer which is used to send.
        """
        self.video_track.clear_sender_buffer()

    async def subscribe_video(self, uid: int) -> None:
        """
        Subscribes to the video of a user.

        Parameters:
            uid: The user ID to subscribe to.
        """
        future = asyncio.Future()

        def callback(
            agora_local_user,
            channel,
            user_id,
            old_state,
            new_state,
            elapse_since_last_state,
        ):
            if new_state == 3:  # Successfully subscribed
                future.set_result(None)
            # elif new_state == 1:  # Subscription failed
            #     future.set_exception(
            #         Exception(
            #             f"Failed to subscribe {user_id} video: state changed from {old_state} to {new_state}"
            #         )
            #     )

        self.on("video_subscribe_state_changed", callback)
        self.local_user.subscribe_video(uid, VideoSubscriptionOptions())

        try:
            await future
        except Exception as e:
            raise Exception(f"video subscription failed for user {uid}: {str(e)}") from e
        finally:
            self.off("video_subscribe_state_changed", callback)

    async def unsubscribe_video(self, uid: int) -> None:
        """
        Unsubscribes from the video of a user.

        Parameters:
            uid: The user ID to unsubscribe from.
        """
        future = asyncio.Future()

        def callback(
            agora_local_user,
            channel,
            user_id,
            old_state,
            new_state,
            elapse_since_last_state,
        ):
            if new_state == 3:  # Successfully unsubscribed
                future.set_result(None)
            else:  # Failed to unsubscribe
                future.set_exception(
                    Exception(
                        f"Failed to unsubscribe {user_id} video: state changed from {old_state} to {new_state}"
                    )
                )

        self.on("video_subscribe_state_changed", callback)
        self.local_user.unsubscribe_video(uid)

        try:
            await future
        except Exception as e:
            raise Exception(f"video unsubscription failed for user {uid}: {str(e)}") from e
        finally:
            self.off("video_subscribe_state_changed", callback)

    def release(self):
        self.local_user.unregister_local_user_observer()
        if self._param.audio_in_enabled:
            self.local_user.unregister_audio_frame_observer()
        if self._param.audio_out_enabled:
            self.audio_track.set_enabled(0)
            self.local_user.unpublish_audio(self.audio_track)
            self.audio_pcm_data_sender.release()
            self.audio_pcm_data_sender = None
            self.audio_track.release()
            self.audio_track = None
        if self._param.camera_in_enabled:
            self.local_user.unregister_video_frame_observer()
        if self._param.camera_out_enabled:
            self.video_track.set_enabled(0)
            self.local_user.unpublish_video(self.video_track)
            self.video_yuv_data_sender.release()
            self.video_yuv_data_sender = None
            self.video_track.release()
            self.video_track = None

        self.media_node_factory.release()
        self.media_node_factory = None
        self.local_user.release()
        self.local_user = None
        self.channel_event_observer = None
        logging.info("local user's channel released")

    # Event Callback

    def on_connection_state_changed(self, agora_rtc_conn, conn_info, reason):
        # logging.debug(f"conn state {conn_info.state}")
        self.connection_state = conn_info.state

    def on_user_joined(self, agora_rtc_conn, user_id):
        # logging.debug(f"User {user_id} joined")
        self.remote_users[user_id] = True

    def on_user_left(self, agora_rtc_conn, user_id, reason):
        logging.info(f"User({type(user_id)}) {user_id} left reason: {reason}")

    def destory_when_user_left(self, user_id):
        # NOTE: like class obj destory in C++,
        # this will be called when the last emit event (user left) is triggered
        if user_id in self.remote_users:
            self.remote_users.pop(user_id, None)
        if user_id in self.channel_event_observer.audio_streams:
            audio_stream = self.channel_event_observer.audio_streams.pop(user_id, None)
            audio_stream.queue.put_nowait(None)
        if user_id in self.channel_event_observer.video_streams:
            video_stream = self.channel_event_observer.video_streams.pop(user_id, None)
            video_stream.queue.put_nowait(None)

    def on_audio_subscribe_state_changed(
        self,
        agora_local_user: rtc.LocalUser,
        channel: str,
        user_id: str,
        old_state: int,
        new_state: int,
        elapse_since_last_state: int,
    ):
        # logging.debug(f"{agora_local_user} {channel} {user_id} {old_state} {new_state}")
        if new_state == 3:  # Successfully subscribed
            if user_id not in self.channel_event_observer.audio_streams:
                self.channel_event_observer.audio_streams[user_id] = rtc.AudioStream()
                logging.info(
                    f"{channel} {user_id} new AudioStream in audio_streams:{self.channel_event_observer.audio_streams}"
                )

    def on_video_subscribe_state_changed(
        self,
        agora_local_user: rtc.LocalUser,
        channel: str,
        user_id: str,
        old_state: int,
        new_state: int,
        elapse_since_last_state: int,
    ):
        if new_state == 3:  # Successfully subscribed
            if user_id not in self.channel_event_observer.video_streams:
                self.channel_event_observer.video_streams[user_id] = VideoStream()
                logging.info(
                    f"{channel} {user_id} new VideoStream in video_streams:{self.channel_event_observer.video_streams}"
                )

    # Audio out
    async def push_pcm_audio_frame(self, pcm_audio_frame: rtc.PcmAudioFrame) -> None:
        """
        Pushes an audio frame to the channel.

        Parameters:
            frame: The pcm audio frame to be pushed.
        """
        frame = copy.copy(pcm_audio_frame)
        ret = self.audio_pcm_data_sender.send_audio_pcm_data(frame)
        logging.debug(
            f"Pushed audio frame: {ret}, audio frame length: {len(pcm_audio_frame.data) if pcm_audio_frame.data else 0}"
        )
        if ret and ret < 0:
            raise Exception(
                f"Failed to send audio frame: {ret}, audio frame length: {len(pcm_audio_frame.data) if pcm_audio_frame.data else 0}"
            )

    # Video in
    def get_video_frames(self, uid: str) -> VideoStream | None:
        """
        Returns the video frames from the channel.

        Returns:
            AudioStream: The video stream.
        """
        return (
            None
            if self.channel_event_observer.video_streams.get(uid) is None
            else self.channel_event_observer.video_streams.get(uid)
        )

    # Video out
    async def push_yuv_video_frame(self, video_frame: ExternalVideoFrame) -> None:
        """
        Pushes an video frame to the channel.

        Parameters:
            frame: The video frame to be pushed.
        """
        frame = copy.copy(video_frame)
        ret = self.video_yuv_data_sender.send_video_frame(frame)
        logging.debug(
            f"Pushed video frame: {ret}, video frame length: {len(video_frame.buffer) if video_frame.buffer else 0}"
        )
        if ret and ret < 0:
            raise Exception(
                f"Failed to send video frame: {ret}, video frame length: {len(video_frame.buffer if video_frame.buffer else 0)}"
            )


class AgoraCallbacks(BaseModel):
    """async callback, no wait"""

    on_connected: Callable[[rtc.RTCConnection, rtc.RTCConnInfo, int], Awaitable[None]]
    on_connection_state_changed: Callable[
        [rtc.RTCConnection, rtc.RTCConnInfo, int], Awaitable[None]
    ]
    on_connection_failure: Callable[[int], Awaitable[None]]
    on_disconnected: Callable[[rtc.RTCConnection, rtc.RTCConnInfo, int], Awaitable[None]]
    on_error: Callable[[str], Awaitable[None]]
    on_participant_connected: Callable[[rtc.RTCConnection, str], Awaitable[None]]
    on_participant_disconnected: Callable[[rtc.RTCConnection, str, int], Awaitable[None]]
    on_data_received: Callable[[bytes, str], Awaitable[None]]
    on_first_participant_joined: Callable[[str], Awaitable[None]]
    on_audio_subscribe_state_changed: Callable[
        [rtc.LocalUser, str, str, int, int, int], Awaitable[None]
    ]
    on_video_subscribe_state_changed: Callable[
        [rtc.LocalUser, str, str, int, int, int], Awaitable[None]
    ]


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

        logging.debug(f"token_claims: {self._token_claims}")
        self._app_id = self._token_claims.app_id
        self._room_name = self._token_claims.rtc.channel_name
        self._join_name = self._token_claims.rtc.uid
        self._params = params
        self._callbacks = callbacks
        self._loop = loop
        self._service = service if service else AgoraService.init(self._params)
        self._channel = RtcChannel(
            params,
            self._token_claims,
            rtc.RtcOptions(
                channel_name=self._token_claims.rtc.channel_name,
                uid=int(self._token_claims.rtc.uid) if self._token_claims.rtc.uid else 0,
                sample_rate=params.audio_in_sample_rate,
                channels=params.audio_in_channels,
                enable_pcm_dump=params.enable_pcm_dump,
            ),
            self._loop,
            self._service,
        )

        self._joined = False
        self._joining = False
        self._leaving = False

        self._other_participant_has_joined = False

        # wait to sub then notify task
        self._in_task: asyncio.Task = self._loop.create_task(self.in_handle())
        self._curr_subscribe_user: int | None = None

        # audio out
        # local participant audio stream

        # TODO: switch room anchor participant, need know room anchor
        # audio in
        # passive sub
        # self._on_participant_audio_frame_task: asyncio.Task | None = None
        # active sub
        self._in_audio_queue = asyncio.Queue[bytes]()
        self._in_audio_task: asyncio.Task = (
            self._loop.create_task(self.in_audio_handle()) if params.audio_in_enabled else None
        )

        # video in
        # passive sub the participant video frame
        self._on_participant_video_frame_task: asyncio.Task | None = None
        # active sub
        # self._in_video_queue = asyncio.Queue[bytes]()
        # self._in_video_task: asyncio.Task = self._loop.create_task(
        #    self.in_video_handle()) if params.video_in_enabled else None

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
        self._channel.on("video_subscribe_state_changed", self.on_video_subscribe_state_changed)

    def verify_token(self) -> bool:
        try:
            self._token_claims = TokenPaser.parse_claims(self._token)
        except Exception as e:
            logging.warning(f"verfiy {self._token} Exception: {e}")
            return False

        return True

    async def cleanup(self):
        if self._in_task and not self._in_task.cancelled():
            self._in_task.cancel()
            await self._in_task
            # logging.info("Cancelled in_task")

        if self._in_audio_task and not self._in_audio_task.cancelled():
            self._in_audio_task.cancel()
            await self._in_audio_task
            # logging.info("Cancelled in_audio_task")

        if (
            self._on_participant_video_frame_task
            and not self._on_participant_video_frame_task.cancelled()
        ):
            self._on_participant_video_frame_task.cancel()
            await self._on_participant_video_frame_task

    async def _join(self):
        self._joining = True
        participants = self.get_participant_ids()
        logging.info(
            f"{self._join_name} Connecting to {self._room_name},"
            f" current remote participants:{participants}"
        )

        try:
            await self._channel.connect()
        except Exception as e:
            self._joining = False
            self._joined = False
            raise e

        logging.info(
            f"local_participant:{self._channel.uid} joined room(channel): {self._room_name} connection_state: {self._channel.connection_state}"
        )
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

            # Set up audio source and track
            # - for pub/write out to local_participant microphone
            if self._params.audio_out_enabled:
                self._channel.set_local_user_audio_track()
                logging.info("local user audio track enabled")

            # Set up video source and track
            # for pub/write out to local_participant camera
            if self._params.camera_out_enabled:
                self._channel.set_local_user_video_track()
                logging.info("local user video track enabled")

            # Check if there are already participants in the room
            participants = self.get_participant_ids()
            if len(participants) > 0 and not self._other_participant_has_joined:
                logging.info(f"first participant {participants[0]} join")
                self._other_participant_has_joined = True
                # TODO: need check who is 主播,
                # default use the first remote participants[0],
                # or callback params use room to broadcast
                await self._callbacks.on_first_participant_joined(participants[0])

            demo_url = self._params.demo_voice_url
            if self._params.camera_in_enabled or self._params.camera_out_enabled:
                demo_url = self._params.demo_video_url
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
        self._channel.release()
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
            return remote_user
        except asyncio.TimeoutError:
            future.cancel()
        except KeyboardInterrupt:
            future.cancel()
        except Exception as e:
            logging.error(f"Error waiting for remote user: {e}")
            raise

    async def send_signaling_message(self, frame: TransportMessageFrame):
        """
        TODO: send signaling message to remote_participant with rtm
        """
        if not self._joined:
            return

        try:
            pass
        except Exception as e:
            logging.error(f"Error sending data: {e}")

    async def send_message(self, frame: TransportMessageFrame):
        """
        Send a chat message in the chat channel.
        TODO: need use biz msg_id, now just use uuid
        """
        if not self._joined:
            return

        msg_id = uuid.uuid4().hex
        try:
            await self._channel.chat.send_message(
                rtc.ChatMessage(
                    message=json.dumps(frame.message),
                    # need use biz message id
                    msg_id=msg_id,
                )
            )
        except Exception as e:
            logging.error(f"Error sending data: {e}")

    def get_participant_ids(self) -> List[str]:
        return list(self._channel.remote_users.keys())

    async def get_participant_metadata(self, participant_id: str) -> dict:
        participant = self._channel.remote_users.get(participant_id)
        if participant:
            # TODO: get remote_participant_info
            return {}
        return {}

    async def subscribe(self, user_id: str):
        """subscribe audio/video
        Idempotent subscribe
        """
        try:
            if self._params.audio_in_enabled:
                await self._channel.subscribe_audio(user_id)
            if self._params.camera_in_enabled:
                await self._channel.subscribe_video(user_id)
        except Exception as e:
            logging.error(f"Error subscribing user {user_id}: {e}", exc_info=True)

    async def unsubscribe(self, user_id: str):
        """unsubscribe audio/video
        Idempotent unsubscribe, but have some error log
        """
        try:
            if self._params.audio_in_enabled:
                await self._channel.unsubscribe_audio(str(user_id))
        except Exception as e:
            logging.error(
                f"Error unsubscribe_audio user {type(user_id)=} {user_id}: {e}", exc_info=True
            )
        try:
            if self._params.camera_in_enabled:
                await self._channel.unsubscribe_video(user_id)
        except Exception as e:
            logging.error(f"Error unsubscribe_video user {user_id}: {e}", exc_info=True)

    # Event Callback

    def on_participant_connected(self, agora_rtc_conn: rtc.RTCConnection, user_id: str):
        async def participant_connected():
            await self._callbacks.on_participant_connected(agora_rtc_conn, user_id)

        # wait
        # self._loop.run_until_complete(participant_connected())
        # no wait
        self._loop.create_task(participant_connected())

    def on_participant_disconnected(
        self, agora_rtc_conn: rtc.RTCConnection, user_id: str, reason: int
    ):
        async def participant_disconnected():
            # befor destory
            participant_ids = self.get_participant_ids()
            if len(participant_ids) > 0 and participant_ids[0] == user_id:
                await self.unsubscribe(user_id)

            self._channel.destory_when_user_left(user_id)

            # after destory
            participant_ids = self.get_participant_ids()
            if len(participant_ids) == 0:
                self._other_participant_has_joined = False
            await self._callbacks.on_participant_disconnected(agora_rtc_conn, user_id, reason)

        self._loop.create_task(participant_disconnected())

    def on_data_received(
        self,
        agora_local_user: rtc.LocalUser,
        user_id: str,
        stream_id: str,
        data: bytes,
        length: int,
    ):
        logging.info(
            f"{agora_local_user} Received stream({stream_id}) message from {user_id} with ({type(data)})length: {length}"
        )
        self._loop.create_task(self._callbacks.on_data_received(data, user_id))

    def on_connected(
        self, agora_rtc_conn: rtc.RTCConnection, conn_info: rtc.RTCConnInfo, reason: int
    ):
        self._loop.create_task(self._callbacks.on_connected(agora_rtc_conn, conn_info, reason))

    def on_connection_state_changed(
        self, agora_rtc_conn: rtc.RTCConnection, conn_info: rtc.RTCConnInfo, reason: int
    ):
        self._loop.create_task(
            self._callbacks.on_connection_state_changed(agora_rtc_conn, conn_info, reason)
        )

    def on_disconnected(
        self, agora_rtc_conn: rtc.RTCConnection, conn_info: rtc.RTCConnInfo, reason: int
    ):
        self._joined = False
        self._loop.create_task(self._callbacks.on_disconnected(agora_rtc_conn, conn_info, reason))

    def on_connection_failure(
        self, agora_rtc_conn: rtc.RTCConnection, conn_info: rtc.RTCConnInfo, reason: int
    ):
        self._joined = False
        self._loop.create_task(self._callbacks.on_connection_failure(reason))

    def on_audio_subscribe_state_changed(
        self,
        agora_local_user: rtc.LocalUser,
        channel: str,
        user_id: str,
        old_state: int,
        new_state: int,
        elapse_since_last_state: int,
    ):
        self._loop.create_task(
            self._callbacks.on_audio_subscribe_state_changed(
                agora_local_user, channel, user_id, old_state, new_state, elapse_since_last_state
            )
        )

    def on_video_subscribe_state_changed(
        self,
        agora_local_user: rtc.LocalUser,
        channel: str,
        user_id: str,
        old_state: int,
        new_state: int,
        elapse_since_last_state: int,
    ):
        self._loop.create_task(
            self._callbacks.on_video_subscribe_state_changed(
                agora_local_user, channel, user_id, old_state, new_state, elapse_since_last_state
            )
        )

    async def in_handle(self) -> None:
        """
        wait remote user to subscirbe, when no participant join;
        then notify the in audio/video handle to get audio/video stream
        """
        logging.debug("start in_handle")
        try:
            self._curr_subscribe_user = await self.wait_for_remote_user()
            logging.info(
                f"current subscribe user {self._curr_subscribe_user} other_participant_has_joined: {self._other_participant_has_joined}"
            )
            await self.subscribe(self._curr_subscribe_user)
            if not self._other_participant_has_joined:
                self._other_participant_has_joined = True
                await self._callbacks.on_first_participant_joined(self._curr_subscribe_user)
        except asyncio.CancelledError:
            logging.info("Cancelled Audio task")
        except Exception as e:
            logging.error(f"in_handle error:{e}", exc_info=True)
            return

    # Audio in

    async def in_audio_handle(self) -> None:
        """active sub the first participant audio frame"""
        logging.info("start in_audio_handle")
        try:
            # from on_playback_audio_frame_before_mixing callback
            # to get the audio frame published by the remote user
            while (
                self._curr_subscribe_user is None
                or self._channel.get_audio_frames(self._curr_subscribe_user) is None
            ):
                await asyncio.sleep(0.1)

            audio_frames = self._channel.get_audio_frames(self._curr_subscribe_user)
            logging.info(f"start get_audio_frames({self._curr_subscribe_user}): {audio_frames}")
            async for audio_frame in audio_frames:
                await self._in_audio_queue.put((audio_frame, self._curr_subscribe_user))
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

    def capture_participant_audio(
        self,
        participant_id: str,
        callback: Callable[[UserAudioRawFrame], Awaitable[None]],
        sample_rate=None,
        num_channels=None,
    ):
        """passive sub the participant audio frame"""
        # TODO: switch participant audio stream
        pass

    # Audio out

    async def write_raw_audio_frames(self, frame_data: bytes):
        if not self._joined:
            return

        try:
            pcm_audio_frame: rtc.PcmAudioFrame = self._convert_output_audio(frame_data)
            await self._channel.push_pcm_audio_frame(pcm_audio_frame)

            # TODONE: need fix push_audio_frame,
            # the sample_rate is not correct, need use output_sample_rate
            # await self._channel.push_audio_frame(frame_data)
        except Exception as e:
            logging.error(f"Error publishing audio: {e}", exc_info=True)

    def _convert_output_audio(self, audio_data: bytes) -> rtc.PcmAudioFrame:
        bytes_per_sample = SAMPLE_WIDTH  # Assuming 16-bit audio sample_width
        total_samples = len(audio_data) // bytes_per_sample
        samples_per_channel = total_samples // self._params.audio_out_channels

        frame = rtc.PcmAudioFrame()
        frame.data = bytearray(audio_data)
        frame.timestamp = 0
        frame.bytes_per_sample = bytes_per_sample
        frame.samples_per_channel = samples_per_channel
        frame.sample_rate = self._params.audio_out_sample_rate
        frame.number_of_channels = self._params.audio_out_channels

        return frame

    # Camera in

    async def in_video_handle(self) -> None:
        """active sub the first participant video frame"""
        logging.info("start in_video_handle")
        try:
            # from on_frame callback
            # to get the video frame published by the remote user
            while (
                self._curr_subscribe_user is None
                or self._channel.get_video_frames(self._curr_subscribe_user) is None
            ):
                await asyncio.sleep(0.1)

            video_frames = self._channel.get_video_frames(self._curr_subscribe_user)
            logging.info(f"start get_video_frames({self._curr_subscribe_user}): {video_frames}")
            async for video_frame in video_frames:
                await self._in_video_queue.put((video_frame, self._curr_subscribe_user))
                # Yield control to allow other tasks to run
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            logging.info("Cancelled video task")
            return

    def _convert_input_video_image(
        self, participant_id: str, video_frame: VideoFrame, target_color_mode: str = "RGB"
    ) -> UserImageRawFrame | None:
        """Convert input video frame to image
        target_color_mode from PIL.Image.Image convert method mode param

        cpu/gpu binding optimization:)
        - now just use opencv to optimize the performance, when use cpu (c/c++)
        - SIMD Vectorization
        - GPU Acceleration
        - use Numba jit to run fast
        """
        match video_frame.type:
            case const.AGORA_VIDEO_PIXEL_I420:  # default use yuv420
                # too slow use python lib 64 ms with cpu, need use GPU optimized
                # image = convert_I420_to_RGB(video_frame)
                image = convert_I420_to_RGB_with_cv(video_frame)  # cv use c/c++ lib 2 ms with cpu
            # case const.AGORA_VIDEO_PIXEL_RGBA:
            # need to check RGBA from video
            #    image = convert_RGBA_to_RGB(video_frame)
            case const.AGORA_VIDEO_PIXEL_NV21:
                # image = convert_NV21_to_RGB(video_frame)
                image = convert_NV21_to_RGB_optimized(video_frame)
            case const.AGORA_VIDEO_PIXEL_I422:
                # image = convert_I422_to_RGB(video_frame)
                image = convert_I422_to_RGB_with_cv(video_frame)
            case const.AGORA_VIDEO_PIXEL_NV12:
                # image = convert_NV12_to_RGB(video_frame)
                image = convert_NV12_to_RGB_optimized(video_frame)
            case _:
                logging.warning(f"buffer type:{video_frame.type} un support convert")
                return None

        if target_color_mode != "RGB":
            image = image.convert(target_color_mode)

        return UserImageRawFrame(
            user_id=participant_id,
            image=image.tobytes(),
            size=(video_frame.width, video_frame.height),
            mode=target_color_mode,
            format="JPEG",
        )

    def capture_participant_video(
        self,
        participant_id: str,
        callback: Callable[[UserImageRawFrame], Awaitable[None]],
        framerate: int = 30,
        video_source: str = "camera",
        color_format: str = "RGB",
    ):
        """passive sub the participant video frame"""
        video_stream = self._channel.get_video_frames(participant_id)
        if not video_stream:
            logging.error(f"participant_id {participant_id} no video stream")
            return

        self._on_participant_video_frame_task = asyncio.create_task(
            self._async_on_participant_video_frame(
                participant_id, callback, video_stream, color_format
            )
        )

    async def _async_on_participant_video_frame(
        self,
        participant_id: str,
        callback: Callable[[UserImageRawFrame], Awaitable[None]],
        video_stream: VideoStream,
        color_format: str = "RGB",
    ):
        logging.info(
            f"Started capture participant_id:{participant_id} from video stream {video_stream}"
        )
        async for video_frame in video_stream:
            try:
                start = time.time() * 1000
                # covert video yuv frame to image (yuv->rgb)
                image_frame = self._convert_input_video_image(
                    participant_id,
                    video_frame,
                    target_color_mode=color_format,
                )
                logging.debug(f"convert_input_video_image time: {time.time()*1000 - start} ms")

                start = time.time() * 1000
                if asyncio.iscoroutinefunction(callback):
                    await callback(image_frame)
                else:
                    callback(image_frame)
                logging.debug(f"callback time: {time.time()*1000 - start} ms")
            except asyncio.CancelledError:
                logging.info("task cancelled")
                break
            except Exception as e:
                logging.error(f"task Error: {e}", exc_info=True)

    # Camera out

    async def write_frame_to_camera(self, frame: ImageRawFrame):
        """
        publish image frame to camera
        !NOTE: (now just support RGBA color_format/mode)
        """
        try:
            image = PIL.Image.frombytes(frame.mode, frame.size, frame.image).convert("RGBA")
            # image.save(os.path.join(VIDEOS_DIR, "temp.png"))

            frame_buffer = bytearray(image.tobytes())
            logging.debug(
                f"width:{image.width}, height:{image.height}"
                f" rgba_len:{image.width * image.height * 4}, len:{len(frame_buffer)}"
            )

            video_frame = ExternalVideoFrame()
            video_frame.type = const.AGORA_VIDEO_BUFFER_RAW_DATA
            video_frame.format = const.AGORA_VIDEO_PIXEL_RGBA
            video_frame.buffer = frame_buffer
            video_frame.stride = image.width
            video_frame.height = image.height
            video_frame.timestamp = 0
            video_frame.metadata = bytearray(b"from achatbot ImageRawFrame")

            await self._channel.push_yuv_video_frame(video_frame)
        except Exception as e:
            logging.error(f"task Error: {e}", exc_info=True)
