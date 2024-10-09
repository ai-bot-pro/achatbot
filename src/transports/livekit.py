import asyncio
import os
from typing import List, Optional
import logging

from apipeline.frames.control_frames import EndFrame
from apipeline.processors.frame_processor import FrameDirection
from apipeline.frames.data_frames import AudioRawFrame


from src.processors.livekit_input_transport_processor import LivekitInputTransportProcessor
from src.processors.livekit_output_transport_processor import LivekitOutputTransportProcessor
from src.common.types import LivekitParams
from src.services.livekit_client import LivekitCallbacks, LivekitTransportClient
from src.transports.base import BaseTransport
from src.types.frames.data_frames import LivekitTransportMessageFrame


class LivekitTransport(BaseTransport):
    def __init__(
        self,
        token: str,
        room_name: str,
        websocket_url: Optional[str] = None,
        params: LivekitParams = LivekitParams(),
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        super().__init__(input_name=input_name, output_name=output_name, loop=loop)

        self._register_event_handler("on_connected")
        self._register_event_handler("on_disconnected")
        self._register_event_handler("on_participant_connected")
        self._register_event_handler("on_participant_disconnected")
        self._register_event_handler("on_audio_track_subscribed")
        self._register_event_handler("on_audio_track_unsubscribed")
        self._register_event_handler("on_video_track_subscribed")
        self._register_event_handler("on_video_track_unsubscribed")
        self._register_event_handler("on_data_received")
        self._register_event_handler("on_first_participant_joined")
        self._register_event_handler("on_participant_left")
        self._register_event_handler("on_call_state_updated")
        logging.info(f"LivekitTransport register event names: {self.event_names}")
        callbacks = LivekitCallbacks(
            on_connected=self._on_connected,
            on_disconnected=self._on_disconnected,
            on_participant_connected=self._on_participant_connected,
            on_participant_disconnected=self._on_participant_disconnected,
            on_audio_track_subscribed=self._on_audio_track_subscribed,
            on_audio_track_unsubscribed=self._on_audio_track_unsubscribed,
            on_video_track_subscribed=self._on_video_track_subscribed,
            on_video_track_unsubscribed=self._on_video_track_unsubscribed,
            on_data_received=self._on_data_received,
            on_first_participant_joined=self._on_first_participant_joined,
        )

        self._params = params
        self._params.websocket_url = websocket_url or os.getenv("LIVEKIT_URL")
        self._params.api_key = params.api_key or os.getenv("LIVEKIT_API_KEY")
        self._params.api_secret = params.api_secret or os.getenv("LIVEKIT_API_SECRET")
        self._client = LivekitTransportClient(
            token, room_name, self._params, callbacks, self._loop,
        )

        self._input: LivekitInputTransportProcessor | None = None
        self._output: LivekitOutputTransportProcessor | None = None

    def input_processor(self) -> LivekitInputTransportProcessor:
        if not self._input:
            self._input = LivekitInputTransportProcessor(
                self._client, self._params, name=self._input_name)
        return self._input

    def output_processor(self) -> LivekitOutputTransportProcessor:
        if not self._output:
            self._output = LivekitOutputTransportProcessor(
                self._client, self._params, name=self._output_name
            )
        return self._output

    @property
    def participant_id(self) -> str:
        return self._client.participant_id

    async def send_audio(self, frame: AudioRawFrame):
        if self._output:
            await self._output.process_frame(frame, FrameDirection.DOWNSTREAM)

    def get_participants(self) -> List[str]:
        return self._client.get_participants()

    async def get_participant_metadata(self, participant_id: str) -> dict:
        return await self._client.get_participant_metadata(participant_id)

    async def set_metadata(self, metadata: str):
        await self._client.set_participant_metadata(metadata)

    async def mute_participant(self, participant_id: str):
        await self._client.mute_participant(participant_id)

    async def unmute_participant(self, participant_id: str):
        await self._client.unmute_participant(participant_id)

    async def _on_connected(self):
        await self._call_event_handler("on_connected")

    async def _on_disconnected(self):
        await self._call_event_handler("on_disconnected")
        # Attempt to reconnect
        try:
            await self._client.connect()
            await self._call_event_handler("on_connected")
        except Exception as e:
            logging.error(f"Failed to reconnect: {e}")

    async def _on_participant_connected(self, participant_id: str):
        await self._call_event_handler("on_participant_connected", participant_id)

    async def _on_participant_disconnected(self, participant_id: str):
        await self._call_event_handler("on_participant_disconnected", participant_id)
        await self._call_event_handler("on_participant_left", participant_id, "disconnected")
        if self._input:
            await self._input.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)
        if self._output:
            await self._output.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)

    async def _on_audio_track_subscribed(self, participant_id: str):
        await self._call_event_handler("on_audio_track_subscribed", participant_id)

    async def _on_audio_track_unsubscribed(self, participant_id: str):
        await self._call_event_handler("on_audio_track_unsubscribed", participant_id)

    async def _on_video_track_subscribed(self, participant_id: str):
        await self._call_event_handler("on_video_track_subscribed", participant_id)

    async def _on_video_track_unsubscribed(self, participant_id: str):
        await self._call_event_handler("on_video_track_unsubscribed", participant_id)

    async def _on_data_received(self, data: bytes, participant_id: str):
        if self._input:
            await self._input.push_app_message(data.decode(), participant_id)
        await self._call_event_handler("on_data_received", data, participant_id)

    async def send_message(self, message: str, participant_id: str | None = None):
        if self._output:
            frame = LivekitTransportMessageFrame(message=message, participant_id=participant_id)
            await self._output.send_message(frame)

    async def cleanup(self):
        if self._input:
            await self._input.cleanup()
        if self._output:
            await self._output.cleanup()
        await self._client.leave()

    async def on_room_event(self, event):
        # Handle room events
        pass

    async def on_participant_event(self, event):
        # Handle participant events
        pass

    async def on_track_event(self, event):
        # Handle track events
        pass

    async def _on_call_state_updated(self, state: str):
        await self._call_event_handler("on_call_state_updated", self, state)

    async def _on_first_participant_joined(self, participant_id: str):
        await self._call_event_handler("on_first_participant_joined", participant_id)
