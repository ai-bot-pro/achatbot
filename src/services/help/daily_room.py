import os
import time
import logging

from src.common.const import *
from src.common.types import DailyRoomArgs, GeneralRoomInfo
from src.common.interface import IRoomManager
from src.common.factory import EngineClass
from src.services.help.daily_rest import (
    DailyRESTHelper,
    DailyRoomObject,
    DailyRoomParams,
    DailyRoomProperties,
)


from dotenv import load_dotenv

load_dotenv(override=True)


class DailyRoom(EngineClass, IRoomManager):
    TAG = "daily_room"

    def __init__(self, **kwargs) -> None:
        self.args = DailyRoomArgs(**kwargs)
        # Create a Daily rest helper
        self.daily_rest_helper = DailyRESTHelper(
            os.getenv("DAILY_API_KEY", ""), os.getenv("DAILY_API_URL", "https://api.daily.co/v1")
        )

    async def close_session(self):
        # http rest api shot session, don't do anything
        pass

    async def create_room(
        self, room_name: str | None = None, exp_time_s: int = ROOM_EXPIRE_TIME
    ) -> GeneralRoomInfo:
        room: DailyRoomObject | None = None
        if not room_name:
            return await self.create_random_room(exp_time_s=exp_time_s)
        try:
            room = self.daily_rest_helper.get_room_from_name(room_name)
        except Exception as ex:
            logging.info(
                f"Failed to get room {room_name} from Daily REST API: {ex}, to new a room: {room_name}"
            )
            # Create a new room
            try:
                params = DailyRoomParams(
                    name=room_name,
                    privacy=self.args.privacy,
                    properties=DailyRoomProperties(
                        exp=time.time() + exp_time_s,
                    ),
                )
                room = self.daily_rest_helper.create_room(params=params)
            except Exception as e:
                raise Exception(f"{e}")

        logging.debug(f"room:{room}")
        return GeneralRoomInfo(
            sid=room.id,
            name=room.name,
            url=room.url,
            creation_time=room.created_at,
            extra_data=room.config.model_dump(),
        )

    async def create_random_room(
        self, exp_time_s: int = RANDOM_ROOM_EXPIRE_TIME
    ) -> GeneralRoomInfo:
        # Create a new random name room
        room: DailyRoomObject | None = None
        try:
            params = DailyRoomParams(
                privacy=self.args.privacy,
                properties=DailyRoomProperties(
                    exp=time.time() + exp_time_s,
                ),
            )
            room = self.daily_rest_helper.create_room(params=params)
        except Exception as e:
            raise Exception(f"{e}")

        logging.debug(f"room:{room}")
        return GeneralRoomInfo(
            sid=room.id,
            name=room.name,
            url=room.url,
            creation_time=room.created_at,
            extra_data=room.config.model_dump(),
        )

    async def gen_token(self, room_name: str, exp_time_s: int = ROOM_TOKEN_EXPIRE_TIME) -> str:
        token = self.daily_rest_helper.get_token_by_name(room_name, exp_time_s)
        logging.debug(f"token:{token}")
        return token

    async def get_room(self, room_name: str) -> GeneralRoomInfo:
        try:
            room = self.daily_rest_helper.get_room_from_name(room_name)
            logging.debug(f"room:{room}")
            g_room = GeneralRoomInfo(
                sid=room.id,
                name=room.name,
                url=room.url,
                creation_time=room.created_at,
                extra_data=room.config.model_dump(),
            )
            return g_room
        except Exception as ex:
            logging.warning(f"Failed to get room {room_name} from Daily REST API: {ex}")
            return None

    async def check_valid_room(self, room_name: str, token: str) -> bool:
        if not room_name:
            return False

        if self.args.privacy == "private":
            try:
                if not token:
                    return False
                token = self.daily_rest_helper.verify_token(token)
                logging.debug(f"token:{token}")
            except Exception as ex:
                logging.warning(f"{token} verify Exception: {ex}")
                return False

        room = await self.get_room(room_name)
        if not room:
            return False

        return True
