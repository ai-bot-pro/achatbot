import logging
import asyncio
from abc import abstractmethod

from apipeline.frames.control_frames import EndFrame

from src.common.types import DailyRoomBotArgs
from src.common.interface import IBot
from src.common.register import Register
from src.common.session import Session
from src.common.types import SessionCtx

register_daily_room_bots = Register('daily-room-bots')


class DailyRoomBot(IBot):

    def __init__(self, **args) -> None:
        self.args = DailyRoomBotArgs(**args)
        if self.args.bot_name is None or len(self.args.bot_name) == 0:
            self.args.bot_name = self.__class__.__name__

        self.task = None
        self._bot_config = self.args.bot_config
        self.session = Session(**SessionCtx("").__dict__)

    def bot_config(self):
        return self._bot_config

    def run(self):
        try:
            asyncio.run(self.arun())
        except KeyboardInterrupt:
            logging.warning("Ctrl-C detected. Exiting!")

    async def arun(self):
        pass

    async def on_first_participant_joined(self, transport, participant):
        self.session.set_client_id(participant['id'])
        logging.info(f"First participant {participant['id']} joined")

    async def on_participant_left(self, transport, participant, reason):
        if self.task is not None:
            await self.task.queue_frame(EndFrame())
        logging.info("Partcipant left. Exiting.")

    async def on_call_state_updated(self, transport, state):
        logging.info("Call state %s " % state)
        if state == "left" and self.task is not None:
            await self.task.queue_frame(EndFrame())
