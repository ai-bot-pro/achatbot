#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""
Daily REST Helpers

Methods that wrap the Daily API to create rooms, check room URLs, and get meeting tokens.

"""

# TODO: use async aiohttp @weedge
import requests
import time

from urllib.parse import urlparse

from pydantic import Field, BaseModel, ValidationError
from typing import Literal, Optional


class DailyRoomSipParams(BaseModel):
    display_name: str = "sw-sip-dialin"
    video: bool = False
    sip_mode: str = "dial-in"
    num_endpoints: int = 1


class DailyRoomProperties(BaseModel, extra="allow"):
    exp: float = Field(default_factory=lambda: time.time() + 5 * 60)
    enable_chat: bool = False
    enable_emoji_reactions: bool = False
    eject_at_room_exp: bool = True
    enable_prejoin_ui: bool = False  # Important for the bot to be able to join headlessly
    enable_dialout: Optional[bool] = None
    sip: Optional[DailyRoomSipParams] = None
    sip_uri: Optional[dict] = None

    @property
    def sip_endpoint(self) -> str:
        if not self.sip_uri:
            return ""
        else:
            return "sip:%s" % self.sip_uri["endpoint"]


class DailyRoomParams(BaseModel):
    name: Optional[str] = None
    privacy: Literal["private", "public"] = "public"
    properties: DailyRoomProperties = DailyRoomProperties()


class DailyRoomObject(BaseModel):
    id: str
    name: str
    api_created: bool
    privacy: str
    url: str
    created_at: str
    config: DailyRoomProperties


class TokenObject(BaseModel):
    exp: int
    room_name: str
    is_owner: bool
    nbf: Optional[int] = None
    enable_screenshare: Optional[bool] = None
    enable_recording: Optional[str] = None


class DailyRESTHelper:
    def __init__(self, daily_api_key: str, daily_api_url: str = "https://api.daily.co/v1"):
        self.daily_api_key = daily_api_key
        self.daily_api_url = daily_api_url

    @staticmethod
    def get_name_from_url(room_url: str) -> str:
        return urlparse(room_url).path[1:]

    def create_room(self, params: DailyRoomParams) -> DailyRoomObject:
        res = requests.post(
            f"{self.daily_api_url}/rooms",
            headers={"Authorization": f"Bearer {self.daily_api_key}"},
            json={**params.model_dump(exclude_none=True)},
        )

        if res.status_code != 200:
            raise Exception(f"Unable to create room: {res.text}")

        data = res.json()

        try:
            room = DailyRoomObject(**data)
        except ValidationError as e:
            raise Exception(f"Invalid response: {e}")

        return room

    def get_room_from_name(self, room_name: str) -> DailyRoomObject:
        res: requests.Response = requests.get(
            f"{self.daily_api_url}/rooms/{room_name}",
            headers={"Authorization": f"Bearer {self.daily_api_key}"},
        )

        if res.status_code != 200:
            raise Exception(f"Room not found: {room_name}")

        data = res.json()

        try:
            room = DailyRoomObject(**data)
        except ValidationError as e:
            raise Exception(f"Invalid response: {e}")

        return room

    def get_room_from_url(
        self,
        room_url: str,
    ) -> DailyRoomObject:
        room_name = DailyRESTHelper.get_name_from_url(room_url)
        return self.get_room_from_name(room_name)

    def get_token_by_url(
        self, room_url: str, expiry_time: float = 60 * 60, owner: bool = True
    ) -> str:
        if not room_url:
            raise Exception(
                "No Daily room specified. You must specify a Daily room in order a token to be generated."
            )

        room_name = DailyRESTHelper.get_name_from_url(room_url)

        return self.get_token_by_name(room_name, expiry_time=expiry_time, owner=owner)

    def get_token_by_name(
        self, room_name: str, expiry_time: float = 60 * 60, owner: bool = True
    ) -> str:
        expiration: float = time.time() + expiry_time
        res: requests.Response = requests.post(
            f"{self.daily_api_url}/meeting-tokens",
            headers={"Authorization": f"Bearer {self.daily_api_key}"},
            json={"properties": {"room_name": room_name, "is_owner": owner, "exp": expiration}},
        )

        if res.status_code != 200:
            raise Exception(f"Failed to create meeting token: {res.status_code} {res.text}")

        token: str = res.json()["token"]

        return token

    def verify_token(self, token: str) -> TokenObject:
        if not token:
            return False

        res: requests.Response = requests.get(
            f"{self.daily_api_url}/meeting-tokens/{token}",
            headers={"Authorization": f"Bearer {self.daily_api_key}"},
        )

        if res.status_code != 200:
            raise Exception(f"Token not found: {token}")

        data = res.json()

        try:
            print("data--->", data)
            token = TokenObject(**data)
        except ValidationError as e:
            raise Exception(f"Invalid response: {e}")

        return token
