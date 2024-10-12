import atexit
import os
import json
import logging
import argparse
import asyncio

from src.common.logger import Logger
from src.common.types import CONFIG_DIR
from src.cmd.bots import import_bots
from src.cmd.bots.run import BotTaskManager, BotTaskRunnerBE, RunBotInfo

from dotenv import load_dotenv
load_dotenv(override=True)


Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)

bot_task_mgr = BotTaskManager()
atexit.register(bot_task_mgr.cleanup)

if __name__ == "__main__":
    """
    python -m src.cmd.bots.main -f config/bots/dummy_bot.json
    python -m src.cmd.bots.main -f config/bots/daily_rtvi_general_bot.json
    python -m src.cmd.bots.main -f config/bots/daily_describe_vision_bot.json
    """
    parser = argparse.ArgumentParser(description="Chat Bot")
    parser.add_argument("-u", type=str, default="", help="Room URL")
    parser.add_argument("-t", type=str, default="", help="Token")
    parser.add_argument(
        "-f", type=str,
        default=os.path.join(CONFIG_DIR, "bots/dummy_bot.json"),
        help="Bot configuration json file")
    args = parser.parse_args()

    bot_config = {}
    with open(args.f, 'r') as f:
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
    try:
        task_runner = BotTaskRunnerBE(bot_task_mgr, **vars(bot_info))
        asyncio.get_event_loop().run_until_complete(task_runner.run())
    except KeyboardInterrupt:
        logging.warning("Ctrl-C detected. Exiting!")
    except Exception as e:
        detail = f"bot {bot_info.chat_bot_name} failed to start process: {e}"
        logging.error(detail, exc_info=True)
