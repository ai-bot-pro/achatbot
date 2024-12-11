import atexit
import multiprocessing
import os
import json
import logging
import argparse
import asyncio

from src.common.logger import Logger
from src.common.types import CONFIG_DIR
from src.cmd.bots.run import BotTaskRunnerBE, RunBotInfo
from src.common.task_manager import TaskManagerFactory

from dotenv import load_dotenv

load_dotenv(override=True)


Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)


if __name__ == "__main__":
    """
    python -m src.cmd.bots.main -f config/bots/dummy_bot.json
    python -m src.cmd.bots.main -f config/bots/daily_rtvi_general_bot.json
    python -m src.cmd.bots.main -f config/bots/daily_describe_vision_bot.json

    python -m src.cmd.bots.main -f config/bots/dummy_bot.json --task_type asyncio
    python -m src.cmd.bots.main -f config/bots/dummy_bot.json --task_type threading
    """
    if os.getenv("ACHATBOT_WORKER_MULTIPROC_METHOD", "fork") == "spawn":
        multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Chat Bot")
    parser.add_argument("--task_type", type=str, default="multiprocessing", help="task type")
    parser.add_argument("--task_done_timeout", type=int, default=5, help="task done timeout s")
    parser.add_argument("-u", type=str, default="", help="Room URL")
    parser.add_argument("-t", type=str, default="", help="Token")
    parser.add_argument("-wsp", type=int, default=0, help="WebSocket Port")
    parser.add_argument("-wsh", type=str, default="", help="WebSocket Host")
    parser.add_argument(
        "-f",
        type=str,
        default=os.path.join(CONFIG_DIR, "bots/dummy_bot.json"),
        help="Bot configuration json file",
    )
    args = parser.parse_args()

    TaskManagerFactory.loop = asyncio.get_event_loop()
    # Bot task dict for status reporting and concurrency control
    bot_task_mgr = TaskManagerFactory.task_manager(
        type=os.getenv("ACHATBOT_TASK_TYPE") or args.task_type,
        task_done_timeout=int(os.getenv("ACHATBOT_TASK_DONE_TIMEOUT") or 0)
        or args.task_done_timeout,
    )
    atexit.register(lambda: TaskManagerFactory.cleanup(bot_task_mgr.cleanup))

    bot_config = {}
    with open(args.f, "r") as f:
        bot_config = json.load(f)
        print(json.dumps(bot_config, indent=4, sort_keys=True))
    bot_info = RunBotInfo(**bot_config)
    logging.info(f"bot_config:{bot_config}")

    room_url = bot_info.room_url
    if len(args.u) > 0:
        bot_info.room_url = args.u
    token = bot_info.token
    if len(args.t) > 0:
        bot_info.token = args.t
    if args.wsp > 0:
        bot_info.websocket_server_port = args.wsp
    token = bot_info.token
    if len(args.wsh) > 0:
        bot_info.websocket_server_host = args.wsh

    try:
        task_runner = BotTaskRunnerBE(bot_task_mgr, **vars(bot_info))
        TaskManagerFactory.loop.run_until_complete(task_runner.run())
    except KeyboardInterrupt:
        logging.warning("Ctrl-C detected. Exiting!")
    except Exception as e:
        detail = f"bot {bot_info.chat_bot_name} failed to start process: {e}"
        logging.error(detail, exc_info=True)
