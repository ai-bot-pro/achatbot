import os
import logging
import asyncio
from typing import Any

import aiohttp
from apipeline.frames.data_frames import AudioRawFrame, ImageRawFrame
from apipeline.processors.frame_processor import FrameDirection, FrameProcessor

from src.services.daily_client import DailyCallbacks, DailyTransportClient
from src.types.frames.data_frames import (
    InterimTranscriptionFrame,
    SpriteFrame,
    TranscriptionFrame,
)
from src.common.types import DailyParams
from src.transports.base import BaseTransport
from src.processors.daily_input_transport_processor import DailyInputTransportProcessor
from src.processors.daily_output_transport_processor import DailyOutputTransportProcessor
from src.types.speech.language import Language


class DailyTransport(BaseTransport):
    def __init__(
        self,
        room_url: str,
        token: str | None,
        bot_name: str,
        params: DailyParams,
        input_name: str | None = None,
        output_name: str | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        super().__init__(input_name=input_name, output_name=output_name, loop=loop)

        # Register supported handlers.
        # User will only be able to register these handlers.
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
        self._register_event_handler("on_participant_updated")
        logging.info(f"DailyTransport register event names: {self.event_names}")
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
            on_participant_updated=self._on_participant_updated,
        )

        self._params = params
        self._params.api_key = os.getenv("DAILY_API_KEY")
        self._client = DailyTransportClient(
            room_url, token, bot_name, self._params, callbacks, self._loop
        )

        self._input: DailyInputTransportProcessor | None = None
        self._output: DailyOutputTransportProcessor | None = None

    #
    # BaseTransport
    #

    def input_processor(self) -> FrameProcessor:
        if not self._input:
            self._input = DailyInputTransportProcessor(
                self._client, self._params, name=self._input_name
            )
        return self._input

    def output_processor(self) -> FrameProcessor:
        if not self._output:
            self._output = DailyOutputTransportProcessor(
                self._client, self._params, name=self._output_name
            )
        return self._output

    #
    # DailyTransport
    #

    @property
    def participant_id(self) -> str:
        return self._client.participant_id

    @property
    def client(self) -> str:
        return self._client

    @property
    def params(self) -> str:
        return self._params

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
            participant_id, self._on_transcription_message
        )

    def capture_participant_video(
        self,
        participant_id: str,
        framerate: int = 30,
        video_source: str = "camera",
        color_format: str = "RGB",
    ):
        if self._input:
            self._input.capture_participant_video(
                participant_id, framerate, video_source, color_format
            )

    async def _on_joined(self, data):
        await self._call_event_handler("on_joined", data)

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
                "Content-Type": "application/json",
            }
            data = {
                "callId": self._params.dialin_settings.call_id,
                "callDomain": self._params.dialin_settings.call_domain,
                "sipUri": sip_endpoint,
            }

            url = f"{self._params.api_url}/dialin/pinlessCallUpdate"

            try:
                async with session.post(url, headers=headers, json=data, timeout=10) as r:
                    if r.status != 200:
                        text = await r.text()
                        logging.error(
                            f"Unable to handle dialin-ready event (status: {r.status}, error: {text})"
                        )
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

    async def _on_participant_updated(self, participant):
        await self._call_event_handler("on_participant_updated", participant)

    async def _on_first_participant_joined(self, participant):
        await self._call_event_handler("on_first_participant_joined", participant)

    async def _on_transcription_message(self, participant_id, message):
        logging.debug(f"participant_id: {participant_id} Received transcription message: {message}")
        text = message["text"]
        timestamp = message["timestamp"]
        is_final = message["rawResponse"]["is_final"]
        try:
            language = message["rawResponse"]["channel"]["alternatives"][0]["languages"][0]
            language = Language(language)
        except KeyError:
            language = None
        if is_final:
            frame = TranscriptionFrame(text, participant_id, timestamp, language)
            logging.debug(f"Transcription (from: {participant_id}): [{text}]")
        else:
            frame = InterimTranscriptionFrame(text, participant_id, timestamp, language)

        if self._input:
            await self._input.push_transcription_frame(frame)
