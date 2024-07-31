import logging
import asyncio
from abc import abstractmethod

from src.common.types import DailyRoomBotArgs
from src.common.interface import IBot
from src.common.register import Register

register_daily_room_bots = Register('daily-room-bots')


class DailyRoomBot(IBot):

    def __init__(self, **args) -> None:
        self.args = DailyRoomBotArgs(**args)
        if self.args.bot_name is None or len(self.args.bot_name) == 0:
            self.args.bot_name = self.__class__.__name__

        self.task = None
        self._bot_config = self.args.bot_config

    def bot_config(self):
        return self._bot_config

    def run(self):
        try:
            asyncio.run(self._run())
        except KeyboardInterrupt:
            logging.warning("Ctrl-C detected. Exiting!")

    @abstractmethod
    async def _run(self):
        pass
