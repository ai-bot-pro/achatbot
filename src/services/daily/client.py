import os
import logging
import asyncio
from typing import Any, Awaitable, Callable, Mapping
from concurrent.futures import ThreadPoolExecutor

from daily import (
    CallClient,
    Daily,
    EventHandler,
    VirtualCameraDevice,
    VirtualMicrophoneDevice,
    VirtualSpeakerDevice)
from pydantic.main import BaseModel
import aiohttp

from apipeline.frames.data_frames import AudioRawFrame, ImageRawFrame
from apipeline.processors.frame_processor import FrameDirection, FrameProcessor
from src.processors.daily_input_transport_processor import DailyInputTransportProcessor
from src.processors.daily_output_transport_processor import DailyOutputTransportProcessor
from src.common.event import EventHandlerManager
from src.types.frames.data_frames import (
    InterimTranscriptionFrame,
    SpriteFrame,
    TranscriptionFrame,
)
from src.common.types import DailyParams, DailyTransportMessageFrame


class DailyCallbacks(BaseModel):
    on_joined: Callable[[Mapping[str, Any]], Awaitable[None]]
    on_left: Callable[[], Awaitable[None]]
    on_error: Callable[[str], Awaitable[None]]
    on_app_message: Callable[[Any, str], Awaitable[None]]
    on_call_state_updated: Callable[[str], Awaitable[None]]
    on_dialin_ready: Callable[[str], Awaitable[None]]
    on_dialout_answered: Callable[[Any], Awaitable[None]]
    on_dialout_connected: Callable[[Any], Awaitable[None]]
    on_dialout_stopped: Callable[[Any], Awaitable[None]]
    on_dialout_error: Callable[[Any], Awaitable[None]]
    on_dialout_warning: Callable[[Any], Awaitable[None]]
    on_first_participant_joined: Callable[[Mapping[str, Any]], Awaitable[None]]
    on_participant_joined: Callable[[Mapping[str, Any]], Awaitable[None]]
    on_participant_left: Callable[[Mapping[str, Any], str], Awaitable[None]]


def completion_callback(future):
    def _callback(*args):
        if not future.cancelled():
            future.get_loop().call_soon_threadsafe(future.set_result, *args)
    return _callback


class DailyTransportClient(EventHandler):

    _daily_initialized: bool = False

    # This is necessary to override EventHandler's __new__ method.
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(
            self,
            room_url: str,
            token: str | None,
            bot_name: str,
            params: DailyParams,
            callbacks: DailyCallbacks,
            loop: asyncio.AbstractEventLoop):
        super().__init__()

        if not self._daily_initialized:
            self._daily_initialized = True
            Daily.init()

        self._room_url: str = room_url
        self._token: str | None = token
        self._bot_name: str = bot_name
        self._params: DailyParams = params
        self._callbacks = callbacks
        self._loop = loop

        self._participant_id: str = ""
        self._video_renderers = {}
        self._transcription_renderers = {}
        self._other_participant_has_joined = False

        self._joined = False
        self._joining = False
        self._leaving = False

        self._executor = ThreadPoolExecutor(max_workers=5)

        self._client: CallClient = CallClient(event_handler=self)

        self._camera: VirtualCameraDevice = Daily.create_camera_device(
            "camera",
            width=self._params.camera_out_width,
            height=self._params.camera_out_height,
            color_format=self._params.camera_out_color_format)

        self._mic: VirtualMicrophoneDevice = Daily.create_microphone_device(
            "mic",
            sample_rate=self._params.audio_out_sample_rate,
            channels=self._params.audio_out_channels,
            non_blocking=True)

        self._speaker: VirtualSpeakerDevice = Daily.create_speaker_device(
            "speaker",
            sample_rate=self._params.audio_in_sample_rate,
            channels=self._params.audio_in_channels,
            non_blocking=True)
        Daily.select_speaker_device("speaker")

    @property
    def participant_id(self) -> str:
        return self._participant_id

    def set_callbacks(self, callbacks: DailyCallbacks):
        self._callbacks = callbacks

    async def send_message(self, frame: DailyTransportMessageFrame):
        future = self._loop.create_future()
        self._client.send_app_message(
            frame.message,
            frame.participant_id,
            completion=completion_callback(future))
        await future

    async def read_next_audio_frame(self) -> AudioRawFrame | None:
        sample_rate = self._params.audio_in_sample_rate
        num_channels = self._params.audio_in_channels
        num_frames = int(sample_rate / 100) * 2  # 20ms of audio

        future = self._loop.create_future()
        self._speaker.read_frames(num_frames, completion=completion_callback(future))
        audio = await future

        if len(audio) > 0:
            return AudioRawFrame(audio=audio, sample_rate=sample_rate, num_channels=num_channels)
        else:
            # If we don't read any audio it could be there's no participant
            # connected. daily-python will return immediately if that's the
            # case, so let's sleep for a little bit (i.e. busy wait).
            await asyncio.sleep(0.01)
            return None

    async def write_raw_audio_frames(self, frames: bytes):
        future = self._loop.create_future()
        self._mic.write_frames(frames, completion=completion_callback(future))
        await future

    async def write_frame_to_camera(self, frame: ImageRawFrame):
        self._camera.write_frame(frame.image)

    async def join(self):
        # Transport already joined, ignore.
        if self._joined or self._joining:
            return

        logging.info(f"Joining {self._room_url}")

        self._joining = True

        # For performance reasons, never subscribe to video streams (unless a
        # video renderer is registered).
        self._client.update_subscription_profiles({
            "base": {
                "camera": "unsubscribed",
                "screenVideo": "unsubscribed"
            }
        })

        self._client.set_user_name(self._bot_name)

        try:
            (data, error) = await self._join()

            if not error:
                self._joined = True
                self._joining = False

                logging.info(f"Joined {self._room_url}")

                if self._token and self._params.transcription_enabled:
                    logging.info(
                        f"Enabling transcription with settings {self._params.transcription_settings}")
                    self._client.start_transcription(
                        self._params.transcription_settings.model_dump())

                await self._callbacks.on_joined(data["participants"]["local"])
            else:
                error_msg = f"Error joining {self._room_url}: {error}"
                logging.error(error_msg)
                await self._callbacks.on_error(error_msg)
        except asyncio.TimeoutError:
            error_msg = f"Time out joining {self._room_url}"
            logging.error(error_msg)
            await self._callbacks.on_error(error_msg)

    async def _join(self):
        future = self._loop.create_future()

        def handle_join_response(data, error):
            if not future.cancelled():
                future.get_loop().call_soon_threadsafe(future.set_result, (data, error))

        self._client.join(
            self._room_url,
            self._token,
            completion=handle_join_response,
            client_settings={
                "inputs": {
                    "camera": {
                        "isEnabled": self._params.camera_out_enabled,
                        "settings": {
                            "deviceId": "camera",
                        },
                    },
                    "microphone": {
                        "isEnabled": self._params.audio_out_enabled,
                        "settings": {
                            "deviceId": "mic",
                            "customConstraints": {
                                "autoGainControl": {"exact": False},
                                "echoCancellation": {"exact": False},
                                "noiseSuppression": {"exact": False},
                            },
                        },
                    },
                },
                "publishing": {
                    "camera": {
                        "sendSettings": {
                            "maxQuality": "low",
                            "encodings": {
                                "low": {
                                    "maxBitrate": self._params.camera_out_bitrate,
                                    "maxFramerate": self._params.camera_out_framerate,
                                }
                            },
                        }
                    }
                },
            })

        return await asyncio.wait_for(future, timeout=10)

    async def leave(self):
        # Transport not joined, ignore.
        if not self._joined or self._leaving:
            return

        self._joined = False
        self._leaving = True

        logging.info(f"Leaving {self._room_url}")

        if self._params.transcription_enabled:
            self._client.stop_transcription()

        try:
            error = await self._leave()
            if not error:
                self._leaving = False
                logging.info(f"Left {self._room_url}")
                await self._callbacks.on_left()
            else:
                error_msg = f"Error leaving {self._room_url}: {error}"
                logging.error(error_msg)
                await self._callbacks.on_error(error_msg)
        except asyncio.TimeoutError:
            error_msg = f"Time out leaving {self._room_url}"
            logging.error(error_msg)
            await self._callbacks.on_error(error_msg)

    async def _leave(self):
        future = self._loop.create_future()

        def handle_leave_response(error):
            if not future.cancelled():
                future.get_loop().call_soon_threadsafe(future.set_result, error)

        self._client.leave(completion=handle_leave_response)

        return await asyncio.wait_for(future, timeout=10)

    async def cleanup(self):
        await self._loop.run_in_executor(self._executor, self._cleanup)

    def _cleanup(self):
        if self._client:
            self._client.release()
            self._client = None

    def participants(self):
        return self._client.participants()

    def participant_counts(self):
        return self._client.participant_counts()

    def start_dialout(self, settings):
        self._client.start_dialout(settings)

    def stop_dialout(self, participant_id):
        self._client.stop_dialout(participant_id)

    def start_recording(self, streaming_settings, stream_id, force_new):
        self._client.start_recording(streaming_settings, stream_id, force_new)

    def stop_recording(self, stream_id):
        self._client.stop_recording(stream_id)

    def capture_participant_transcription(self, participant_id: str, callback: Callable):
        if not self._params.transcription_enabled:
            return

        self._transcription_renderers[participant_id] = callback

    def capture_participant_video(
            self,
            participant_id: str,
            callback: Callable,
            framerate: int = 30,
            video_source: str = "camera",
            color_format: str = "RGB"):
        # Only enable camera subscription on this participant
        self._client.update_subscriptions(participant_settings={
            participant_id: {
                "media": "subscribed"
            }
        })

        self._video_renderers[participant_id] = callback

        self._client.set_video_renderer(
            participant_id,
            self._video_frame_received,
            video_source=video_source,
            color_format=color_format)

    #
    #
    # Daily (EventHandler)
    #

    def on_app_message(self, message: Any, sender: str):
        self._call_async_callback(self._callbacks.on_app_message, message, sender)

    def on_call_state_updated(self, state: str):
        self._call_async_callback(self._callbacks.on_call_state_updated, state)

    def on_dialin_ready(self, sip_endpoint: str):
        self._call_async_callback(self._callbacks.on_dialin_ready, sip_endpoint)

    def on_dialout_answered(self, data: Any):
        self._call_async_callback(self._callbacks.on_dialout_answered, data)

    def on_dialout_connected(self, data: Any):
        self._call_async_callback(self._callbacks.on_dialout_connected, data)

    def on_dialout_stopped(self, data: Any):
        self._call_async_callback(self._callbacks.on_dialout_stopped, data)

    def on_dialout_error(self, data: Any):
        self._call_async_callback(self._callbacks.on_dialout_error, data)

    def on_dialout_warning(self, data: Any):
        self._call_async_callback(self._callbacks.on_dialout_warning, data)

    def on_participant_joined(self, participant):
        id = participant["id"]
        logging.info(f"Participant joined {id}")

        if not self._other_participant_has_joined:
            self._other_participant_has_joined = True
            self._call_async_callback(self._callbacks.on_first_participant_joined, participant)

        self._call_async_callback(self._callbacks.on_participant_joined, participant)

    def on_participant_left(self, participant, reason):
        id = participant["id"]
        logging.info(f"Participant left {id}")

        self._call_async_callback(self._callbacks.on_participant_left, participant, reason)

    def on_transcription_message(self, message: Mapping[str, Any]):
        participant_id = ""
        if "participantId" in message:
            participant_id = message["participantId"]

        if participant_id in self._transcription_renderers:
            callback = self._transcription_renderers[participant_id]
            self._call_async_callback(callback, participant_id, message)

    def on_transcription_error(self, message):
        logging.error(f"Transcription error: {message}")

    def on_transcription_started(self, status):
        logging.debug(f"Transcription started: {status}")

    def on_transcription_stopped(self, stopped_by, stopped_by_error):
        logging.debug("Transcription stopped")

    #
    # Daily (CallClient callbacks)
    #

    def _video_frame_received(self, participant_id, video_frame):
        callback = self._video_renderers[participant_id]
        self._call_async_callback(
            callback,
            participant_id,
            video_frame.buffer,
            (video_frame.width,
             video_frame.height),
            video_frame.color_format)

    def _call_async_callback(self, callback, *args):
        future = asyncio.run_coroutine_threadsafe(callback(*args), self._loop)
        future.result()


class DailyTransport(EventHandlerManager):

    def __init__(
            self,
            room_url: str,
            token: str | None,
            bot_name: str,
            params: DailyParams,
            input_name: str | None = None,
            output_name: str | None = None,
            loop: asyncio.AbstractEventLoop | None = None):
        super().__init__(input_name=input_name, output_name=output_name, loop=loop)

        callbacks = DailyCallbacks(
            on_joined=self._on_joined,
            on_left=self._on_left,
            on_error=self._on_error,
            on_app_message=self._on_app_message,
            on_call_state_updated=self._on_call_state_updated,
            on_dialin_ready=self._on_dialin_ready,
            on_dialout_answered=self._on_dialout_answered,
            on_dialout_connected=self._on_dialout_connected,
            on_dialout_stopped=self._on_dialout_stopped,
            on_dialout_error=self._on_dialout_error,
            on_dialout_warning=self._on_dialout_warning,
            on_first_participant_joined=self._on_first_participant_joined,
            on_participant_joined=self._on_participant_joined,
            on_participant_left=self._on_participant_left,
        )
        self._params = params
        self._params.api_key = os.getenv("DAILY_API_KEY")

        self._client = DailyTransportClient(
            room_url, token, bot_name, params, callbacks, self._loop)
        self._input: DailyInputTransportProcessor | None = None
        self._output: DailyOutputTransportProcessor | None = None

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_joined")
        self._register_event_handler("on_left")
        self._register_event_handler("on_app_message")
        self._register_event_handler("on_call_state_updated")
        self._register_event_handler("on_dialin_ready")
        self._register_event_handler("on_dialout_answered")
        self._register_event_handler("on_dialout_connected")
        self._register_event_handler("on_dialout_stopped")
        self._register_event_handler("on_dialout_error")
        self._register_event_handler("on_dialout_warning")
        self._register_event_handler("on_first_participant_joined")
        self._register_event_handler("on_participant_joined")
        self._register_event_handler("on_participant_left")

    def input_processor(self) -> FrameProcessor:
        if not self._input:
            self._input = DailyInputTransportProcessor(
                self._client, self._params, name=self._input_name)
        return self._input

    def output_processor(self) -> FrameProcessor:
        if not self._output:
            self._output = DailyOutputTransportProcessor(
                self._client, self._params, name=self._output_name)
        return self._output

    def input_audio_stream(self):
        pass

    def output_audio_stream(self):
        pass

    #
    # DailyTransport
    #

    @ property
    def participant_id(self) -> str:
        return self._client.participant_id

    async def send_image(self, frame: ImageRawFrame | SpriteFrame):
        if self._output:
            await self._output.process_frame(frame, FrameDirection.DOWNSTREAM)

    async def send_audio(self, frame: AudioRawFrame):
        if self._output:
            await self._output.process_frame(frame, FrameDirection.DOWNSTREAM)

    def participants(self):
        return self._client.participants()

    def participant_counts(self):
        return self._client.participant_counts()

    def start_dialout(self, settings=None):
        self._client.start_dialout(settings)

    def stop_dialout(self, participant_id):
        self._client.stop_dialout(participant_id)

    def start_recording(self, streaming_settings=None, stream_id=None, force_new=None):
        self._client.start_recording(streaming_settings, stream_id, force_new)

    def stop_recording(self, stream_id=None):
        self._client.stop_recording(stream_id)

    def capture_participant_transcription(self, participant_id: str):
        self._client.capture_participant_transcription(
            participant_id,
            self._on_transcription_message
        )

    def capture_participant_video(
            self,
            participant_id: str,
            framerate: int = 30,
            video_source: str = "camera",
            color_format: str = "RGB"):
        if self._input:
            self._input.capture_participant_video(
                participant_id, framerate, video_source, color_format)

    async def _on_joined(self, participant):
        await self._call_event_handler("on_joined", participant)

    async def _on_left(self):
        await self._call_event_handler("on_left")

    async def _on_error(self, error):
        # TODO(aleix): Report error to input/output transports. The one managing
        # the client should report the error.
        pass

    async def _on_app_message(self, message: Any, sender: str):
        if self._input:
            await self._input.push_app_message(message, sender)
        await self._call_event_handler("on_app_message", message, sender)

    async def _on_call_state_updated(self, state: str):
        await self._call_event_handler("on_call_state_updated", state)

    async def _handle_dialin_ready(self, sip_endpoint: str):
        if not self._params.dialin_settings:
            return

        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self._params.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "callId": self._params.dialin_settings.call_id,
                "callDomain": self._params.dialin_settings.call_domain,
                "sipUri": sip_endpoint
            }

            url = f"{self._params.api_url}/dialin/pinlessCallUpdate"

            try:
                async with session.post(url, headers=headers, json=data, timeout=10) as r:
                    if r.status != 200:
                        text = await r.text()
                        logging.error(
                            f"Unable to handle dialin-ready event (status: {r.status}, error: {text})")
                        return

                    logging.debug("Event dialin-ready was handled successfully")
            except asyncio.TimeoutError:
                logging.error(f"Timeout handling dialin-ready event ({url})")
            except BaseException as e:
                logging.error(f"Error handling dialin-ready event ({url}): {e}")

    async def _on_dialin_ready(self, sip_endpoint):
        if self._params.dialin_settings:
            await self._handle_dialin_ready(sip_endpoint)
        await self._call_event_handler("on_dialin_ready", sip_endpoint)

    async def _on_dialout_answered(self, data):
        await self._call_event_handler("on_dialout_answered", data)

    async def _on_dialout_connected(self, data):
        await self._call_event_handler("on_dialout_connected", data)

    async def _on_dialout_stopped(self, data):
        await self._call_event_handler("on_dialout_stopped", data)

    async def _on_dialout_error(self, data):
        await self._call_event_handler("on_dialout_error", data)

    async def _on_dialout_warning(self, data):
        await self._call_event_handler("on_dialout_warning", data)

    async def _on_participant_joined(self, participant):
        await self._call_event_handler("on_participant_joined", participant)

    async def _on_participant_left(self, participant, reason):
        await self._call_event_handler("on_participant_left", participant, reason)

    async def _on_first_participant_joined(self, participant):
        await self._call_event_handler("on_first_participant_joined", participant)

    async def _on_transcription_message(self, participant_id, message):
        text = message["text"]
        timestamp = message["timestamp"]
        is_final = message["rawResponse"]["is_final"]
        if is_final:
            frame = TranscriptionFrame(text, participant_id, timestamp)
            logging.debug(f"Transcription (from: {participant_id}): [{text}]")
        else:
            frame = InterimTranscriptionFrame(text, participant_id, timestamp)

        if self._input:
            await self._input.push_transcription_frame(frame)
