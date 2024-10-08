import os
import time
import logging

from src.common.interface import IRoom


from dotenv import load_dotenv
load_dotenv(override=True)


class LivekitRoom(IRoom):
    ROOM_EXPIRE_TIME = 30 * 60  # 30 minutes
    ROOM_TOKEN_EXPIRE_TIME = 30 * 60  # 30 minutes
    RANDOM_ROOM_EXPIRE_TIME = 5 * 60  # 5 minutes
    RANDOM_ROOM_TOKEN_EXPIRE_TIME = 5 * 60  # 5 minutes

    DAILYLANGCHAINRAGBOT_EXPIRE_TIME = 25 * 60

    def __init__(self) -> None:
        pass

    def create_room(self, room_name, exp_time_s: int = ROOM_EXPIRE_TIME):
        room = None

        return room

    def token(self, exp_time_s: int = ROOM_TOKEN_EXPIRE_TIME) -> str:
        token = ""
        return token

    def get_room(self, room_name):
        room = None

        return room
