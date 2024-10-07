import os
from livekit import api
import asyncio


async def create_room():
    # will automatically use theLIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET env vars
    lkapi = api.LiveKitAPI()
    room_info = await lkapi.room.create_room(
        api.CreateRoomRequest(name=os.getenv("ROOM_NAME", "chat-room")),
    )
    print(room_info)
    results = await lkapi.room.list_rooms(api.ListRoomsRequest())
    print(results)
    await lkapi.aclose()

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(create_room())
