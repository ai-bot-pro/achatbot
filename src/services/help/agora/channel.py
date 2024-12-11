import os
import logging
import uuid

from src.common.const import *
from src.common.types import AgoraChannelArgs, GeneralRoomInfo
from src.common.interface import IRoomManager
from src.common.factory import EngineClass
from src.services.help.agora.token import TokenPaser

from agora_realtime_ai_api.token_builder.realtimekit_token_builder import RealtimekitTokenBuilder

from dotenv import load_dotenv

load_dotenv(override=True)


class AgoraChannel(EngineClass, IRoomManager):
    """
    type: Rtc (base)
    no room, just use channel name as room name
    """

    TAG = "agora_rtc_channel"

    def __init__(self, **kwargs) -> None:
        self.args = AgoraChannelArgs(**kwargs)
        self.app_id = os.environ.get("AGORA_APP_ID")
        self.app_cert = os.environ.get("AGORA_APP_CERT")
        if not self.app_id:
            raise ValueError("AGORA_APP_ID must be set in the environment.")

    async def close_session(self):
        # no http rest api session, don't do anything
        pass

    async def create_room(
        self, room_name: str | None = None, exp_time_s: int = ROOM_EXPIRE_TIME
    ) -> GeneralRoomInfo:
        if not room_name:
            return await self.create_random_room(exp_time_s=exp_time_s)

        return GeneralRoomInfo(
            name=room_name,
        )

    async def create_random_room(
        self, exp_time_s: int = RANDOM_ROOM_EXPIRE_TIME
    ) -> GeneralRoomInfo:
        return GeneralRoomInfo(
            name=str(uuid.uuid4()),
        )

    async def gen_token(self, room_name: str, exp_time_s: int = ROOM_TOKEN_EXPIRE_TIME) -> str:
        token = RealtimekitTokenBuilder.build_token(
            self.app_id, self.app_cert, room_name, 0, expiration_in_seconds=exp_time_s
        )
        logging.debug(f"token:{token}")
        return token

    async def get_room(self, room_name: str) -> GeneralRoomInfo:
        g_room = GeneralRoomInfo(
            name=room_name,
        )
        return g_room

    async def check_valid_room(self, room_name: str, token: str) -> bool:
        if not room_name or not token:
            return False

        try:
            res = TokenPaser.is_expired(token)
            if res is False:
                return False
        except Exception as e:
            logging.error(f"check_valid_room error: {e}")
            return False

        room = await self.get_room(room_name)
        if not room:
            return False

        return True
