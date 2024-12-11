import asyncio
import logging
import os
from signal import SIGINT, SIGTERM

import numpy as np
from livekit import rtc, api


# bytes("weedge-newbee") this is our shared key, it must match the one used by your clients
# https://ultra-terminal-re8nmd.sandbox.livekit.io/?tab=demo
# e.g.: https://ultra-terminal-re8nmd.sandbox.livekit.io/rooms/chat-room#weedge-newbee
SHARED_KEY = b"weedge-newbee"

WIDTH, HEIGHT = 1280, 720


async def draw_cube(source: rtc.VideoSource):
    MID_W, MID_H = 640, 360
    cube_size = 60
    vertices = (
        np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ]
        )
        * cube_size
    )
    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]

    arr = np.zeros((WIDTH * HEIGHT * 4), dtype=np.uint8)
    angle = 0

    while True:
        start_time = asyncio.get_event_loop().time()
        arr.fill(0)
        rot = np.dot(
            np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(angle), -np.sin(angle)],
                    [0, np.sin(angle), np.cos(angle)],
                ]
            ),
            np.array(
                [
                    [np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)],
                ]
            ),
        )
        proj_points = [
            [int(pt[0] / (pt[2] / 200 + 1)), int(pt[1] / (pt[2] / 200 + 1))]
            for pt in np.dot(vertices, rot)
        ]

        for e in edges:
            x1, y1, x2, y2 = *proj_points[e[0]], *proj_points[e[1]]
            for t in np.linspace(0, 1, 100):
                x, y = (
                    int(MID_W + (1 - t) * x1 + t * x2),
                    int(MID_H + (1 - t) * y1 + t * y2),
                )
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if 0 <= x + dx < WIDTH and 0 <= y + dy < HEIGHT:
                            idx = (y + dy) * WIDTH * 4 + (x + dx) * 4
                            arr[idx : idx + 4] = [255, 255, 255, 255]

        f = rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.ARGB, arr)
        source.capture_frame(f)
        angle += 0.02

        code_duration = asyncio.get_event_loop().time() - start_time
        await asyncio.sleep(1 / 30 - code_duration)


async def main(room: rtc.Room):
    @room.on("e2ee_state_changed")
    def on_e2ee_state_changed(participant: rtc.Participant, state: rtc.EncryptionState) -> None:
        logging.info("e2ee state changed: %s %s", participant.identity, state)

    url = os.getenv("LIVEKIT_URL")
    logging.info("connecting to %s", url)
    # ensure LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET are set
    token = (
        api.AccessToken()
        .with_identity("python-e2ee")
        .with_name("Python E2EE")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=os.getenv("ROOM_NAME", "chat-room"),
            )
        )
        .to_jwt()
    )
    url = os.getenv("LIVEKIT_URL")
    try:
        e2ee_options = rtc.E2EEOptions()
        e2ee_options.key_provider_options.shared_key = SHARED_KEY

        await room.connect(
            url,
            token,
            options=rtc.RoomOptions(
                auto_subscribe=True,
                e2ee=e2ee_options,
            ),
        )

        logging.info("connected to room %s with e2ee %s", room.name, e2ee_options)
    except rtc.ConnectError as e:
        logging.error("failed to connect to the room: %s", e)
        return False

    # publish a track
    source = rtc.VideoSource(WIDTH, HEIGHT)
    track = rtc.LocalVideoTrack.create_video_track("cube", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_CAMERA
    publication = await room.local_participant.publish_track(track, options)
    logging.info("published track %s", publication.sid)

    asyncio.ensure_future(draw_cube(source))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            # logging.FileHandler("e2ee.log"),
            logging.StreamHandler(),
        ],
    )

    loop = asyncio.get_event_loop()
    room = rtc.Room(loop=loop)

    async def cleanup():
        await room.disconnect()
        loop.stop()

    asyncio.ensure_future(main(room))
    for signal in [SIGINT, SIGTERM]:
        loop.add_signal_handler(signal, lambda: asyncio.ensure_future(cleanup()))

    try:
        loop.run_forever()
    finally:
        loop.close()
