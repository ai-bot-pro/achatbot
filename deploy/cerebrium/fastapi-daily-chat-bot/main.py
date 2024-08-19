from achatbot.cmd.http.server import fastapi_daily_bot_serve as serve
from achatbot.cmd.http.server.fastapi_daily_bot_serve import BotInfo

from cerebrium import get_secret


def run(param_1: str, param_2: str, run_id):  # run_id is optional, injected by Cerebrium at runtime
    my_results = {"1": param_1, "2": param_2}
    my_status_code = 200  # if you want to return a specific status code

    return {"my_result": my_results, "status_code": my_status_code}  # return your results

def register_bot(bot_name: str = "DummyBot"):
    return serve.register_bot(bot_name)


async def bot_join(chat_bot_name: str, info: BotInfo):
    res = await serve.bot_join(chat_bot_name, info)
    return res


async def bot_join_room(room_name: str, chat_bot_name: str, info: BotInfo):
    res = await serve.bot_join_room(room_name, chat_bot_name, info)
    return res


async def create_random_room():
    res = await serve.create_random_room()
    return res


async def get_status(pid: int):
    res = await serve.get_status(pid)
    return res
