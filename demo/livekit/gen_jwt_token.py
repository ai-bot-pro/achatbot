import datetime
import os
import uuid
from livekit import api


def generate_token():
    user_identity = str(uuid.uuid4())

    # will automatically use the LIVEKIT_API_KEY and LIVEKIT_API_SECRET env vars
    token = (
        api.AccessToken()
        .with_identity(user_identity)
        .with_name("weedge")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=os.getenv("ROOM_NAME", "chat-room")
            )
        )
        .with_ttl(datetime.timedelta(hours=1))
        .to_jwt()
    )
    return token


if __name__ == "__main__":
    token = generate_token()
    print(token)
