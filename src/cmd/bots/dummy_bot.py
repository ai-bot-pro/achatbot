import os
import logging
import asyncio

from src.cmd.bots.base import AIBot
from src.common.interface import IBot
from . import register_ai_room_bots


@register_ai_room_bots.register
class DummyBot(AIBot):
    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()

    async def arun(self):
        logging.info(f"Starting bot,env: {os.environ}")
        logging.info(f"dummy bot run with args: {self.args}")
        await asyncio.sleep(10)
        logging.info("dummy bot over")
