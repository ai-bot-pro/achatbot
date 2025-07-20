import uuid
import logging
from typing import Optional

from pydantic import BaseModel

from src.common.task_manager.base import TaskManager
from src.services.help import RoomManagerEnvInit
from src.common.const import *
from src.common.session import Session
from src.common.connector import ConnectorInit
from src.common.interface import IBot, IConnector, IRoomManager
from src.common.types import GeneralRoomInfo, BotRunArgs, SessionCtx
from src.cmd.bots import BotInfo, import_bots, import_websocket_bots, register_ai_room_bots


class EngineClassInfo(BaseModel):
    tag: Optional[str] = None
    args: Optional[dict] = None


class RunBotInfo(BotInfo):
    task_connector: Optional[EngineClassInfo] = None
    room_manager: Optional[EngineClassInfo] = None


class BotTaskRunner:
    def __init__(self, task_mgr: TaskManager | None = None, **kwargs):
        if task_mgr is None:
            raise ValueError("task_mgr is None")
        self.task_mgr = task_mgr
        self.run_bot_info = RunBotInfo(**kwargs)
        logging.info(f"run_bot_info: {self.run_bot_info}")

        self.task_connector: IConnector = None
        if self.run_bot_info.task_connector:
            self.task_connector = ConnectorInit.getEngine(
                self.run_bot_info.task_connector.tag, **self.run_bot_info.task_connector.args
            )

        self.room_mgr: IRoomManager = None
        if self.run_bot_info.room_manager:
            self.room_mgr = RoomManagerEnvInit.initEngine(
                self.run_bot_info.room_manager.tag, self.run_bot_info.room_manager.args
            )
        else:
            self.room_mgr = RoomManagerEnvInit.initEngine()

        self.session = Session(**SessionCtx(uuid.uuid4()).__dict__)
        self._pid = 0
        self._bot_obj: IBot | None = None

    async def _run_room_bot(self, bot_info: BotInfo):
        room_name = bot_info.room_name
        room_url = bot_info.room_url
        bot_token = bot_info.token
        if not self.room_mgr:
            logging.error("need init RoomManager!")
            return

        is_valid = await self.room_mgr.check_valid_room(bot_info.room_name, bot_info.token)
        if not is_valid:
            room: GeneralRoomInfo = await self.room_mgr.create_room(
                bot_info.room_name, exp_time_s=ROOM_EXPIRE_TIME
            )
            bot_token = await self.room_mgr.gen_token(room.name, ROOM_EXPIRE_TIME)
            await self.room_mgr.close_session()
            room_url = room.url
            room_name = room.name

        kwargs = BotRunArgs(
            room_name=room_name,
            room_url=room_url,
            token=bot_token,
            bot_name=bot_info.chat_bot_name,
            bot_config=bot_info.config,
            bot_config_list=bot_info.config_list,
            services=bot_info.services,
        ).__dict__
        self._bot_obj: IBot = register_ai_room_bots[bot_info.chat_bot_name](**kwargs)
        logging.info(f"bot {bot_info.chat_bot_name} loading")
        self._bot_obj.load()
        logging.info(f"bot {bot_info.chat_bot_name} load done")

        logging.info(f"bot {bot_info.chat_bot_name} starting with pid {self._pid}")
        if self.run_bot_info.is_background is True:
            self._pid = await self.task_mgr.run_task(
                self._bot_obj.run, bot_info.chat_bot_name, bot_info.room_name
            )
        else:
            await self._bot_obj.async_run()
        logging.info(f"bot {bot_info.chat_bot_name} started with pid {self._pid}")

    async def _run_websocket_bot(self, bot_info: BotInfo):
        kwargs = BotRunArgs(
            bot_name=bot_info.chat_bot_name,
            bot_config=bot_info.config,
            bot_config_list=bot_info.config_list,
            services=bot_info.services,
            websocket_server_port=bot_info.websocket_server_port,
            websocket_server_host=bot_info.websocket_server_host,
        ).__dict__
        self._bot_obj: IBot = register_ai_room_bots[bot_info.chat_bot_name](**kwargs)
        self._bot_obj.load()
        self._pid = await self.task_mgr.run_task(
            self._bot_obj.run, bot_info.chat_bot_name, bot_info.room_name
        )

    @property
    def bot_config(self):
        return self._bot_obj.bot_config() if self._bot_obj else {}

    @property
    def pid(self):
        return self._pid if self._pid else None

    async def run_bot(self, bot_info: BotInfo):
        if bot_info.transport_type == "websocket":
            if import_websocket_bots(bot_info.chat_bot_name) is False:
                detail = f"un import bot: {bot_info.chat_bot_name}"
                logging.error(detail)
                return
            await self._run_websocket_bot(bot_info)
        else:
            if import_bots(bot_info.chat_bot_name) is False:
                detail = f"un import bot: {bot_info.chat_bot_name}"
                logging.error(detail)
                return
            await self._run_room_bot(bot_info)

    async def run(self):
        pass


class BotTaskRunnerFE(BotTaskRunner):
    async def run(self):
        # run local bot
        if self.task_connector is None:
            await self.run_bot(self.run_bot_info)
            return

        self.task_connector.send(("RUN_BOT_TASK", self.run_bot_info, self.session), "fe")


class BotTaskRunnerBE(BotTaskRunner):
    async def run(self):
        # run local bot
        if self.task_connector is None:
            await self.run_bot(self.run_bot_info)
            return

        # run remote bot, bot info from connector recv
        # just a easy recv, no done for consumed msg
        # or Idempotent retry with task status
        while True:
            res = self.task_connector.recv("be")
            if res is None:
                continue

            msg = res[0]
            bot_info: RunBotInfo = res[1]
            session: Session = res[2]
            logging.debug(f"msg:{msg}, bot_info:{bot_info}, session:{session}")

            if msg == "STOP":
                logging.info(f"bot {bot_info.chat_bot_name} stopped")
                return

            match msg:
                case "RUN_BOT_TASK":
                    logging.info(f"bot {bot_info.chat_bot_name} running")
                    await self.run_bot(bot_info)
                case _:
                    logging.warning(f"{msg} unsupport")
