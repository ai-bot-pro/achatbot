import os
import logging
import time

from src.common.interface import IBot
from . import register_ai_room_bots


@register_ai_room_bots.register
class DummyBot(IBot):
    def __init__(self, **args) -> None:
        self.args = args

    def run(self):
        logging.info(f"Starting bot,env: {os.environ}")
        logging.info(f"dummy bot run with args: {self.args}")
        time.sleep(10)
        logging.info("dummy bot over")

    async def arun(self):
        pass

    def bot_config(self) -> dict:
        return {}
