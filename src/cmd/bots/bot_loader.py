import asyncio
import json
import logging
import os
import pathlib
from typing import Literal

from src.cmd.bots import import_bots, import_fastapi_websocket_bots, import_websocket_bots
from src.cmd.bots.base import AIBot
from src.common.types import BotRunArgs
from src.cmd.bots.run import RunBotInfo
from src.cmd.bots import register_ai_fastapi_ws_bots, register_ai_room_bots


"""
!TODO: load bot config from remote(config center) or flow config UI
"""


async def load_bot_config_from_local(file_path: str):
    bot_config = {}
    with open(file_path, "r") as f:
        bot_config = json.load(f)
        print(json.dumps(bot_config, indent=4, sort_keys=True))
    bot_info = RunBotInfo(**bot_config)
    logging.info(f"bot_info:{bot_info}")
    return bot_info


class BotLoader:
    """
    load bot at once
    """

    run_bots = {}
    lock = asyncio.Lock()

    @staticmethod
    async def load_bot(
        local_file_path: str | pathlib.PosixPath,
        is_re_init=False,
        bot_type: Literal["room_bot", "ws_bot", "fastapi_ws_bot"] = "room_bot",
    ) -> AIBot:
        """
        load once from str or pathlib.PosixPath(for container volume)
        """
        if isinstance(local_file_path, str) and not os.path.isfile(local_file_path):
            logging.error(f"config_path: {local_file_path} not found")
            raise FileNotFoundError

        if isinstance(local_file_path, pathlib.PosixPath) and not local_file_path.is_file():
            logging.error(f"config_path: {local_file_path} not found")
            raise FileNotFoundError

        bot_info = await load_bot_config_from_local(file_path=local_file_path)
        bot_args = BotRunArgs(
            bot_name=bot_info.chat_bot_name,
            bot_config=bot_info.config,
            bot_config_list=bot_info.config_list,
            services=bot_info.services,
            handle_sigint=bot_info.handle_sigint,
            websocket_server_host=bot_info.websocket_server_host,
            websocket_server_port=bot_info.websocket_server_port,
            room_name=bot_info.room_name,
            room_url=bot_info.room_url,
            token=bot_info.token,
        )

        async with BotLoader.lock:
            if bot_info.chat_bot_name in BotLoader.run_bots and is_re_init is False:
                logging.info(f"{bot_info.chat_bot_name} inited, don't re-init.")
                return BotLoader.run_bots[bot_info.chat_bot_name]

            match bot_type:
                case "room_bot":
                    if import_bots(bot_info.chat_bot_name) is False:
                        detail = f"un import bot: {bot_info.chat_bot_name}"
                        raise Exception(detail)

                    logging.info(f"register bots: {register_ai_room_bots.items()}")
                    if bot_info.chat_bot_name not in register_ai_room_bots:
                        detail = f"bot {bot_info.chat_bot_name} don't exist"
                        raise Exception(detail)

                    run_bot = register_ai_room_bots[bot_info.chat_bot_name](**vars(bot_args))
                case "ws_bot":
                    if import_websocket_bots(bot_info.chat_bot_name) is False:
                        detail = f"un import bot: {bot_info.chat_bot_name}"
                        raise Exception(detail)

                    logging.info(f"register bots: {register_ai_room_bots.items()}")
                    if bot_info.chat_bot_name not in register_ai_room_bots:
                        detail = f"bot {bot_info.chat_bot_name} don't exist"
                        raise Exception(detail)

                    run_bot = register_ai_room_bots[bot_info.chat_bot_name](**vars(bot_args))
                case "fastapi_ws_bot":
                    if import_fastapi_websocket_bots(bot_info.chat_bot_name) is False:
                        detail = f"un import bot: {bot_info.chat_bot_name}"
                        raise Exception(detail)

                    logging.info(f"register bots: {register_ai_fastapi_ws_bots.items()}")
                    if bot_info.chat_bot_name not in register_ai_fastapi_ws_bots:
                        detail = f"bot {bot_info.chat_bot_name} don't exist"
                        raise Exception(detail)

                    run_bot = register_ai_fastapi_ws_bots[bot_info.chat_bot_name](
                        websocket=None, **vars(bot_args)
                    )
            BotLoader.run_bots[bot_info.chat_bot_name] = run_bot

            return run_bot
