import os
import time
import logging

from src.services.help.daily_rest import DailyRESTHelper, DailyRoomObject, DailyRoomParams, DailyRoomProperties


from dotenv import load_dotenv
load_dotenv(override=True)


class DailyRoom():
    ROOM_EXPIRE_TIME = 30 * 60  # 30 minutes
    ROOM_TOKEN_EXPIRE_TIME = 30 * 60  # 30 minutes
    RANDOM_ROOM_EXPIRE_TIME = 5 * 60  # 5 minutes
    RANDOM_ROOM_TOKEN_EXPIRE_TIME = 5 * 60  # 5 minutes

    DAILYLANGCHAINRAGBOT_EXPIRE_TIME = 25 * 60

    def __init__(self) -> None:
        # Create a Daily rest helper
        self.daily_rest_helper = DailyRESTHelper(
            os.getenv("DAILY_API_KEY", ""),
            os.getenv("DAILY_API_URL", "https://api.daily.co/v1"))

    def create_room(self, room_name, exp_time_s: int = ROOM_EXPIRE_TIME):
        try:
            room = self.daily_rest_helper.get_room_from_name(room_name)
        except Exception as ex:
            logging.info(
                f"Failed to get room {room_name} from Daily REST API: {ex}, to new a room: {room_name}")
            # Create a new room
            try:
                params = DailyRoomParams(
                    name=room_name,
                    properties=DailyRoomProperties(
                        exp=time.time() + exp_time_s,
                    ),
                )
                room = self.daily_rest_helper.create_room(params=params)
            except Exception as e:
                raise Exception(f"{e}")

        return room

    def create_random_room(self, exp_time_s: int = RANDOM_ROOM_EXPIRE_TIME):
        # Create a new room
        room: DailyRoomObject | None = None
        params = DailyRoomParams(
            properties=DailyRoomProperties(
                exp=time.time() + exp_time_s,
            ),
        )
        room = self.daily_rest_helper.create_room(params=params)

        return room

    def get_token(self, room_url: str, exp_time_s: int = ROOM_TOKEN_EXPIRE_TIME) -> str:
        user_token = self.daily_rest_helper.get_token(room_url, exp_time_s)
        return user_token

    def get_room(self, room_name):
        room = self.daily_rest_helper.get_room_from_name(room_name)
        return room
