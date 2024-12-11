import asyncio
import datetime
import logging
from signal import SIGINT, SIGTERM
from typing import Union
import os

from livekit import api, rtc, protocol

# https://docs.livekit.io/home/client/events/#Events
# NOTE!!!! please see Room on method Available events Arguments :)


async def main(room: rtc.Room) -> None:
    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
        logging.info("participant connected: %s %s", participant.sid, participant.identity)

    @room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        logging.info("participant disconnected: %s %s", participant.sid, participant.identity)

    @room.on("local_track_published")
    def on_local_track_published(
        publication: rtc.LocalTrackPublication,
        track: Union[rtc.LocalAudioTrack, rtc.LocalVideoTrack],
    ):
        logging.info("local track published: %s", publication.sid)

    @room.on("active_speakers_changed")
    def on_active_speakers_changed(speakers: list[rtc.Participant]):
        logging.info("active speakers changed: %s", speakers)

    @room.on("local_track_unpublished")
    def on_local_track_unpublished(publication: rtc.LocalTrackPublication):
        logging.info("local track unpublished: %s", publication.sid)

    @room.on("track_published")
    def on_track_published(
        publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ):
        logging.info(
            "track published: %s from participant %s (%s)",
            publication.sid,
            participant.sid,
            participant.identity,
        )

    @room.on("track_unpublished")
    def on_track_unpublished(
        publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ):
        logging.info("track unpublished: %s", publication.sid)

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logging.info("track subscribed: %s", publication.sid)
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            print("Subscribed to an Video Track")
            _video_stream = rtc.VideoStream(track)
            # video_stream is an async iterator that yields VideoFrame
        elif track.kind == rtc.TrackKind.KIND_AUDIO:
            print("Subscribed to an Audio Track")
            _audio_stream = rtc.AudioStream(track)
            # audio_stream is an async iterator that yields AudioFrame

    @room.on("track_unsubscribed")
    def on_track_unsubscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logging.info("track unsubscribed: %s", publication.sid)

    @room.on("track_muted")
    def on_track_muted(publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        logging.info("track muted: %s", publication.sid)

    @room.on("track_unmuted")
    def on_track_unmuted(
        publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ):
        logging.info("track unmuted: %s", publication.sid)

    @room.on("data_received")
    def on_data_received(data: rtc.DataPacket):
        logging.info("received data from %s: %s", data.participant.identity, data.data)

    @room.on("connection_quality_changed")
    def on_connection_quality_changed(participant: rtc.Participant, quality: rtc.ConnectionQuality):
        logging.info("connection quality changed for %s", participant.identity)

    @room.on("track_subscription_failed")
    def on_track_subscription_failed(
        participant: rtc.RemoteParticipant, track_sid: str, error: str
    ):
        logging.info("track subscription failed: %s %s", participant.identity, error)

    @room.on("connection_state_changed")
    def on_connection_state_changed(state: rtc.ConnectionState):
        logging.info("connection state changed: %s", state)

    @room.on("connected")
    def on_connected() -> None:
        logging.info("connected")

    @room.on("disconnected")
    def on_disconnected(reason: protocol.models.DisconnectReason) -> None:
        logging.info(f"disconnected reason:{reason}")

    @room.on("reconnecting")
    def on_reconnecting() -> None:
        logging.info("reconnecting")

    @room.on("reconnected")
    def on_reconnected() -> None:
        logging.info("reconnected")

    # will automatically use the LIVEKIT_API_KEY and LIVEKIT_API_SECRET env vars
    token = (
        api.AccessToken()
        .with_identity("python-bot")
        .with_name("Python Bot")
        .with_grants(api.VideoGrants(room_join=True, room=os.getenv("ROOM_NAME", "chat-room")))
        .with_ttl(datetime.timedelta(hours=1))
        .to_jwt()
    )
    # need LIVEKIT_URL env vars to connect
    await room.connect(os.getenv("LIVEKIT_URL"), token)
    logging.info(f"room isconnected:{room.isconnected()}")
    logging.info("local participant: %s", room.local_participant)
    logging.info("connected to room %s", room.name)
    logging.info("participants: %s", room.remote_participants)

    # pub data
    await room.local_participant.publish_data(
        "hello world", destination_identities=[p.sid for p in room.remote_participants.values()]
    )

    # chat
    chat = rtc.ChatManager(room)

    # receiving chat
    @chat.on("message_received")
    def on_message_received(msg: rtc.ChatMessage):
        print(f"message received: {msg.participant.identity}: {msg.message}")

    # send msg
    await chat.send_message("hello world msg")

    # have some disconnect BUG
    # while room.isconnected():
    #    print("room disconnecting")
    #    await asyncio.sleep(1)
    #    await room.disconnect()

    logging.info(f"room isconnected:{room.isconnected()}")


if __name__ == "__main__":
    # golang style

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(funcName)s - %(message)s",
        handlers=[
            # logging.FileHandler("room_rtc_events.log"),
            logging.StreamHandler()
        ],
    )

    loop = asyncio.get_event_loop()
    room = rtc.Room(loop=loop)

    async def cleanup():
        print("isconnected:", room.isconnected())
        try:
            await room.disconnect()
        except AssertionError as err:
            print("AssertionError", err)
        except Exception as e:
            print("Exception", e)
        loop.stop()
        print("cleanup ok")

    asyncio.ensure_future(main(room))
    for signal in [SIGINT, SIGTERM]:
        loop.add_signal_handler(signal, lambda: asyncio.ensure_future(cleanup()))

    try:
        loop.run_forever()
    finally:
        loop.close()
