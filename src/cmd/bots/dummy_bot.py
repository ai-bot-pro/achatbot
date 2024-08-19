import logging
import asyncio

from .base import DailyRoomBot, register_daily_room_bots


@register_daily_room_bots.register
class DummyBot(DailyRoomBot):
    async def arun(self):
        logging.info(f"dummy bot run with args: {self.args}")
        await asyncio.sleep(10)
        logging.info("dummy bot over")
