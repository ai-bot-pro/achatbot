import os
import json
import logging
import argparse

from src.common.interface import IBot
from src.common.logger import Logger
from src.common.types import DailyRoomBotArgs, CONFIG_DIR
from src.cmd.bots import BotInfo, import_bots, register_daily_room_bots

from dotenv import load_dotenv
load_dotenv(override=True)


Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)


def run_daily_bot(bot_info: BotInfo):
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
    bot_obj: IBot = register_daily_room_bots[bot_info.chat_bot_name](**kwargs)
    bot_obj.run()


if __name__ == "__main__":
    """
    python -m src.cmd.bots.main -f config/bots/daily_rtvi_general_bot.json
    python -m src.cmd.bots.main -f config/bots/daily_describe_vision_bot.json
    """
    parser = argparse.ArgumentParser(description="Chat Bot")
    parser.add_argument("-b", type=str, default="daily", help="webRTC: daily")
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
    bot_info = BotInfo(**bot_config)
    logging.info(f"bot_config:{bot_config}")

    if import_bots(bot_info.chat_bot_name) is False:
        detail = f"un import bot: {bot_info.chat_bot_name}"
        logging.error(detail)
        exit()

    room_url = bot_info.room_url
    if len(args.u) > 0:
        bot_info.room_url = args.u
    token = bot_info.token
    if len(args.t) > 0:
        bot_info.token = args.t
    try:
        match args.b:
            case "daily":
                run_daily_bot(bot_info)
            case _:
                run_daily_bot(bot_info)
    except Exception as e:
        detail = f"bot {bot_info.chat_bot_name} failed to start process: {e}"
        raise Exception(detail)
