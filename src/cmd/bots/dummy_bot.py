import os
import logging
import asyncio
import time

from fastapi import WebSocket
from src.cmd.bots.base import AIBot
from src.common.interface import IBot
from . import register_ai_room_bots, register_ai_fastapi_ws_bots


@register_ai_room_bots.register
@register_ai_fastapi_ws_bots.register
class DummyBot(AIBot):
    def __init__(self, **args) -> None:
        self._websocket = args.pop("websocket", None)
        super().__init__(**args)
        self.init_bot_config()

    def set_fastapi_websocket(self, websocket: WebSocket):
        self._websocket = websocket

    async def arun(self):
        logging.info(f"Starting bot,env: {os.environ}")
        logging.info(f"dummy bot run with args: {self.args}")
        # time.sleep(10)
        await asyncio.sleep(10)
        logging.info("dummy bot over")
