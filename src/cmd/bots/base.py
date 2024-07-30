import asyncio

from apipeline.pipeline.runner import PipelineRunner

from src.common.interface import IBot
from src.common.register import Register
register_rtvi_bots = Register('rtvi-chat-bot')


class BaseBot(IBot):

    def __init__(self, room_url, token, bot_config, bot_name=None, **kwargs) -> None:
        self.task = None

    def run(self):
        asyncio.run(PipelineRunner().run(self.task))
