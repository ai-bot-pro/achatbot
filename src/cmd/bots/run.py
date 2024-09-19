import uuid
import logging
import multiprocessing
from typing import Optional

from pydantic import BaseModel

from src.common.session import Session
from src.common.connector import ConnectorInit
from src.common.interface import IBot, IConnector
from src.common.types import DailyRoomBotArgs, SessionCtx
from src.cmd.bots import BotInfo, import_bots, register_daily_room_bots


class ConnectorInfo(BaseModel):
    tag: Optional[str] = None
    args: Optional[dict] = None


class RunBotInfo(BotInfo):
    task_connector: Optional[ConnectorInfo] = None


class BotTaskManager:
    def __init__(self) -> None:
        """
        just use dict to store bot process for local task
        !TODO: @weedge
        - if dist task, need database to storage bot process info
        - shecdule task
        """
        self._bot_procs = {}

    @property
    def bot_procs(self):
        return self._bot_procs

    def run_task(self, target, bot_name: str, tag: str, **kwargs):
        """
        use multiprocessing to run task
        !TODO: @weedge
        - use threading to run task
        - use asyncio create task to run
        """
        bot_process: multiprocessing.Process = multiprocessing.Process(
            target=target, name=bot_name, kwargs=kwargs)
        bot_process.start()
        pid = bot_process.pid
        self._bot_procs[pid] = (bot_process, tag)
        return pid

    def get_bot_proces_num(self, tag: str):
        num = 0
        for val in self._bot_procs.values():
            proc: multiprocessing.Process = val[0]
            _tag = val[1]
            if _tag == tag and proc.is_alive():
                num += 1
        return num

    def get_bot_processor(self, pid):
        if pid in self._bot_procs:
            return self._bot_procs[pid]
        return None

    def cleanup(self):
        # Clean up function, just to be extra safe
        for pid, (proc, tag) in self._bot_procs.items():
            if proc.is_alive():
                proc.join()
                proc.terminate()
                proc.close()
                logging.info(f"pid:{pid} tag:{tag} proc: {proc} close")
            else:
                logging.warning(f"pid:{pid} tag:{tag} proc: {proc} already closed")


class BotTaskRunner:
    def __init__(self, task_mgr: BotTaskManager | None = None, **kwargs):
        if task_mgr is None:
            raise ValueError("task_mgr is None")
        self.task_mgr = task_mgr
        self.run_bot_info = RunBotInfo(**kwargs)
        logging.info(f"run_bot_info: {self.run_bot_info}")
        self.task_connector: IConnector = None
        if self.run_bot_info.task_connector:
            self.task_connector = ConnectorInit.getEngine(
                self.run_bot_info.task_connector.tag,
                **self.run_bot_info.task_connector.args)
        self.session = Session(**SessionCtx(uuid.uuid4()).__dict__)
        self._pid = 0
        self._bot_obj: IBot | None = None

    def run_daily_bot(self, bot_info: BotInfo):
        room_url = bot_info.room_url
        bot_token = bot_info.token
        if len(bot_info.room_url.strip()) == 0:
            from src.services.help.daily_rest import DailyRoomObject
            from src.services.help.daily_room import DailyRoom
            daily_room_obj = DailyRoom()
            room: DailyRoomObject = daily_room_obj.create_room(
                bot_info.room_name, exp_time_s=DailyRoom.ROOM_EXPIRE_TIME)
            bot_token = daily_room_obj.get_token(room.url, DailyRoom.ROOM_EXPIRE_TIME)
            room_url = room.url

        kwargs = DailyRoomBotArgs(
            room_url=room_url,
            token=bot_token,
            bot_name=bot_info.chat_bot_name,
            bot_config=bot_info.config,
            bot_config_list=bot_info.config_list,
            services=bot_info.services,
        ).__dict__
        self._bot_obj = register_daily_room_bots[bot_info.chat_bot_name](**kwargs)

        self._pid = self.task_mgr.run_task(
            self._bot_obj.run, bot_info.chat_bot_name, bot_info.room_name)

    @property
    def bot_config(self):
        return self._bot_obj.bot_config() if self._bot_obj else {}

    @property
    def pid(self):
        return self._pid if self._pid else None

    def run_bot(self, bot_info: BotInfo):
        match bot_info.bot_type:
            case "daily":
                self.run_daily_bot(bot_info)
            case _:
                self.run_daily_bot(bot_info)

    def run(self):
        pass


class BotTaskRunnerFE(BotTaskRunner):
    def run(self):
        # run local bot
        if self.task_connector is None:
            self.run_bot(self.run_bot_info)
            return

        self.task_connector.send(("RUN_BOT_TASK", self.run_bot_info, self.session), 'fe')


class BotTaskRunnerBE(BotTaskRunner):
    def run(self):
        # run local bot
        if self.task_connector is None:
            self.run_bot(self.run_bot_info)
            return

        # run remote bot, bot info from connector recv
        # just a easy recv, no done for consumed msg
        # or Idempotent retry with task status
        while True:
            res = self.task_connector.recv('be')
            if res is None:
                continue

            msg = res[0]
            bot_info: RunBotInfo = res[1]
            session: Session = res[2]
            logging.debug(f"msg:{msg}, bot_info:{bot_info}, session:{session}")

            if msg == "STOP":
                logging.info(f"bot {bot_info.chat_bot_name} stopped")
                return

            if import_bots(bot_info.chat_bot_name) is False:
                detail = f"un import bot: {bot_info.chat_bot_name}"
                logging.error(detail)
                continue

            match msg:
                case "RUN_BOT_TASK":
                    logging.info(f"bot {bot_info.chat_bot_name} running")
                    self.run_bot(bot_info)
                case _:
                    logging.warn(f"{msg} unsupport")
