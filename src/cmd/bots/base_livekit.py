import logging

from livekit import rtc
from apipeline.frames.control_frames import EndFrame

from src.cmd.bots.base import AIRoomBot
from src.transports.livekit import LivekitTransport


class LivekitRoomBot(AIRoomBot):
    def regisiter_room_event(self, transport: LivekitTransport):
        transport.add_event_handler(
            "on_connected",
            self.on_connected,
        )
        transport.add_event_handler(
            "on_error",
            self.on_error,
        )
        transport.add_event_handler(
            "on_connection_state_changed",
            self.on_connection_state_changed,
        )
        transport.add_event_handler(
            "on_disconnected",
            self.on_disconnected,
        )
        transport.add_event_handler(
            "on_participant_connected",
            self.on_participant_connected,
        )
        transport.add_event_handler(
            "on_participant_disconnected",
            self.on_participant_disconnected,
        )
        transport.add_event_handler(
            "on_audio_track_subscribed",
            self.on_audio_track_subscribed,
        )
        transport.add_event_handler(
            "on_audio_track_unsubscribed",
            self.on_audio_track_unsubscribed,
        )
        transport.add_event_handler(
            "on_video_track_subscribed",
            self.on_video_track_subscribed,
        )
        transport.add_event_handler(
            "on_video_track_unsubscribed",
            self.on_video_track_unsubscribed,
        )
        transport.add_event_handler(
            "on_data_received",
            self.on_data_received,
        )
        transport.add_event_handler(
            "on_first_participant_joined",
            self.on_first_participant_joined,
        )

    async def on_connected(
        self,
        transport: LivekitTransport,
        room: rtc.Room,
    ):
        logging.debug(f"on_connected room---->{room}")

    async def on_error(self, transport: LivekitTransport, error_msg: str):
        logging.debug("on_error error_msg---->{error_msg}")

    async def on_connection_state_changed(
        self, transport: LivekitTransport, state: rtc.ConnectionState
    ):
        logging.debug("on_connection_state_changed----> state %s " % state)
        if state == rtc.ConnectionState.CONN_DISCONNECTED:
            await self.task.queue_frame(EndFrame())

    async def on_disconnected(self, transport: LivekitTransport, reason: str):
        logging.info("on_disconnected----> reason %s, Exiting." % reason)
        await self.task.queue_frame(EndFrame())

    async def on_participant_connected(
        self, transport: LivekitTransport, participant: rtc.RemoteParticipant
    ):
        logging.debug(f"on_participant_connected---->{participant}")

    async def on_participant_disconnected(
        self, transport: LivekitTransport, participant: rtc.RemoteParticipant
    ):
        logging.debug(f"Partcipant {participant} left.")
        logging.debug(f"current remote Partcipants {transport.get_participants()}")

    async def on_audio_track_subscribed(
        self, transport: LivekitTransport, participant: rtc.RemoteParticipant
    ):
        logging.debug(f"on_audio_track_subscribed---->{participant}")

    async def on_audio_track_unsubscribed(
        self, transport: LivekitTransport, participant: rtc.RemoteParticipant
    ):
        logging.debug(f"on_audio_track_unsubscribed---->{participant}")

    async def on_video_track_subscribed(
        self, transport: LivekitTransport, participant: rtc.RemoteParticipant
    ):
        logging.debug(f"on_video_track_subscribed---->{participant}")

    async def on_video_track_unsubscribed(
        self, transport: LivekitTransport, participant: rtc.RemoteParticipant
    ):
        logging.debug(f"on_video_track_unsubscribed---->{participant}")

    async def on_data_received(
        self, transport: LivekitTransport, data: bytes, participant: rtc.RemoteParticipant
    ):
        logging.debug(f"on_data_received size ---->{len(data)} from {participant}")

    async def on_first_participant_joined(
        self, transport: LivekitTransport, participant: rtc.RemoteParticipant
    ):
        logging.info(f"on_first_participant_joined---->{participant}")

        # TODO: need know room anchor participant
        # now just support one by one chat with bot
        # need open audio_in_participant_enabled
        transport.capture_participant_audio(
            participant_id=participant.sid,
        )
