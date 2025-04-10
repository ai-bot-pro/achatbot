import datetime
import logging
import uuid

from pydantic import BaseModel

try:
    from livekit import api
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error("In order to use Livekit API, you need to `pip install achatbot[livekit-api]`.")
    raise Exception(f"Missing module: {e}")

from src.common.const import *
from src.common.types import GeneralRoomInfo, LivekitRoomArgs
from src.common.interface import IRoomManager
from src.common.factory import EngineClass


from dotenv import load_dotenv

load_dotenv(override=True)


class LivekitRoom(EngineClass, IRoomManager):
    TAG = "livekit_room"

    def __init__(self, **kwargs) -> None:
        self.args = LivekitRoomArgs(**kwargs)
        # live kit http api with pb(protobuf) serialize data
        self._http_api = api.LiveKitAPI() if self.args.is_common_session else None

    async def close_session(self):
        if self._http_api:
            await self._http_api.aclose()

    async def create_room(
        self, room_name: str | None = None, exp_time_s: int = ROOM_EXPIRE_TIME
    ) -> GeneralRoomInfo:
        name = room_name
        empty_time_s = exp_time_s if exp_time_s else ROOM_EXPIRE_TIME
        if not room_name:
            name = "room_" + str(uuid.uuid4())
            if not exp_time_s:
                empty_time_s = RANDOM_ROOM_EXPIRE_TIME

        http_api = self._http_api if self._http_api else api.LiveKitAPI()
        room = await http_api.room.create_room(
            api.CreateRoomRequest(
                name=name,
                empty_timeout=empty_time_s,
            )
        )
        logging.debug(f"create_room:{room}")

        if not self._http_api:
            await http_api.aclose()

        g_room = GeneralRoomInfo(
            sid=room.sid,
            name=room.name,
            ttl_s=room.empty_timeout,
            creation_time=str(room.creation_time),
            # NOTE: room.__dict__, python3.11.0<=version don't ok,python.3.11.7 ok
            extra_data={
                "enabled_codecs": room.enabled_codecs,
                "departure_timeout": room.departure_timeout,
                "max_participants": room.max_participants,
                "num_participants": room.num_participants,
                "num_publishers": room.num_publishers,
                "active_recording": room.active_recording,
                "version": room.version,
            },
        )

        return g_room

    async def gen_token(self, room_name: str, exp_time_s: int = ROOM_TOKEN_EXPIRE_TIME) -> str:
        user_identity = str(uuid.uuid4())
        token = (
            api.AccessToken()
            .with_identity(user_identity)
            .with_name(self.args.bot_name)
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room_name,
                )
            )
            .with_ttl(datetime.timedelta(seconds=exp_time_s))
            .to_jwt()
        )
        logging.debug(f"token:{token}")

        return token

    async def get_room(self, room_name: str) -> GeneralRoomInfo:
        http_api = self._http_api if self._http_api else api.LiveKitAPI()
        result = await http_api.room.list_rooms(api.ListRoomsRequest(names=[room_name]))

        logging.debug(f"list_rooms:{result.rooms}")
        if len(result.rooms) == 0:
            return None

        if not self._http_api:
            await http_api.aclose()

        room = result.rooms[0]
        g_room = GeneralRoomInfo(
            sid=room.sid,
            name=room.name,
            creation_time=str(room.creation_time),
            extra_data={
                "enabled_codecs": room.enabled_codecs,
                "departure_timeout": room.departure_timeout,
                "max_participants": room.max_participants,
                "num_participants": room.num_participants,
                "num_publishers": room.num_publishers,
                "active_recording": room.active_recording,
                "version": room.version,
            },
        )

        return g_room

    async def check_valid_room(self, room_name: str, token: str) -> bool:
        if not room_name or not token:
            return False

        try:
            api.TokenVerifier().verify(token)
        except Exception as e:
            logging.warning(f"{token} verify Exception: {e}")
            return False

        room = await self.get_room(room_name)
        if not room:
            return False

        return True
