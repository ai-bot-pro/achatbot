import atexit
import logging
import multiprocessing
import os
import argparse
from typing import Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

from src.common.const import *
from src.common.logger import Logger
from src.common.types import GeneralRoomInfo
from src.services.help.daily_room import DailyRoom
from src.cmd.bots import import_bots, register_ai_room_bots
from src.cmd.bots.run import BotTaskManager, BotTaskRunnerFE, RunBotInfo


from dotenv import load_dotenv
load_dotenv(override=True)


Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)


# --------------------- API -----------------

class APIResponse(BaseModel):
    error_code: int = 0
    error_detail: str = ""
    data: Any | None = None


# biz error code
ERROR_CODE_NO_ROOM = 10000
ERROR_CODE_BOT_UN_REGISTER = 10001
ERROR_CODE_BOT_FAIL_TOKEN = 10002
ERROR_CODE_BOT_MAX_LIMIT = 10003
ERROR_CODE_BOT_UN_PROC = 10004

# --------------------- Bot ----------------------------


MAX_BOTS_PER_ROOM = 10


# Bot sub-process dict for status reporting and concurrency control
bot_task_mgr = BotTaskManager()
atexit.register(bot_task_mgr.cleanup)

# ------------ Fast API Config ------------ #

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------ Helper methods ------------ #


def escape_bash_arg(s):
    return "'" + s.replace("'", "'\\''") + "'"


def check_host_whitelist(request: Request):
    host_whitelist = os.getenv("HOST_WHITELIST", "")
    request_host_url = request.headers.get("host")

    if not host_whitelist:
        return True

    # Split host whitelist by comma
    allowed_hosts = host_whitelist.split(",")

    # Return True if no whitelist exists are specified
    if len(allowed_hosts) < 1:
        return True

    # Check for apex and www variants
    if any(domain in allowed_hosts for domain in [request_host_url, f"www.{request_host_url}"]):
        return True

    return False


# ------------ Fast API Routes ------------ #
# https://fastapi.tiangolo.com/async/

@app.middleware("http")
async def allowed_hosts_middleware(request: Request, call_next):
    # Middle that optionally checks for hosts in a whitelist
    if not check_host_whitelist(request):
        raise HTTPException(status_code=403, detail="Host access denied")
    response = await call_next(request)
    return response


def app_status():
    logging.info(f"{os.environ}")
    return APIResponse().model_dump()


@app.get("/health")
async def health():
    return JSONResponse(APIResponse().model_dump())


@app.get("/readiness")
async def readiness():
    # todo
    return JSONResponse(APIResponse().model_dump())


@app.get("/metrics")
async def metrics():
    return JSONResponse(APIResponse().model_dump())


@app.get("/create_room/{name}")
async def create_room(name):
    """create room then redirect to room url"""
    try:
        daily_room_obj = DailyRoom()
        room: GeneralRoomInfo = await daily_room_obj.create_room(
            name, exp_time_s=ROOM_EXPIRE_TIME)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}")

    return RedirectResponse(room.url)


"""
curl -XPOST "http://0.0.0.0:4321/create_random_room" \
    -H "Content-Type: application/json"
"""


@app.post("/create_random_room")
async def fastapi_create_random_room(
        exp_time_s: int = RANDOM_ROOM_EXPIRE_TIME) -> JSONResponse:
    try:
        res = await create_random_room(exp_time_s)
    except Exception as e:
        logging.error(f"Exception in create_random_room: {e}")
        raise HTTPException(status_code=500, detail=f"{e}")
    return JSONResponse(res)


async def create_random_room(exp_time_s: int = RANDOM_ROOM_EXPIRE_TIME) -> dict[str, Any]:
    """create random room and token return"""
    if exp_time_s > MAX_RANDOM_ROOM_EXPIRE_TIME:
        raise HTTPException(
            status_code=400,
            detail=f"exp_time_s must be less than {MAX_RANDOM_ROOM_EXPIRE_TIME} s")
    daily_room_obj = DailyRoom()
    room: GeneralRoomInfo = await daily_room_obj.create_room(exp_time_s=exp_time_s)

    # Give the agent a token to join the session
    bot_token = await daily_room_obj.gen_token(room.name, exp_time_s=exp_time_s)

    if not room or not bot_token:
        detail = f"Failed to get token for room: {room.name}"
        return APIResponse(error_code=ERROR_CODE_BOT_FAIL_TOKEN, error_detail=detail).model_dump()

    data = {"room": room, "token": bot_token}
    return APIResponse(data=data).model_dump()


@app.get("/register_bot/{bot_name}")
def fastapi_register_bot(bot_name: str = "DummyBot") -> JSONResponse:
    try:
        res = register_bot(bot_name)
    except Exception as e:
        logging.error(f"Exception in register_bot: {e}")
        raise HTTPException(status_code=500, detail=f"{e}")
    return JSONResponse(res)


def register_bot(bot_name: str = "DummyBot") -> dict[str, Any]:
    """register bot, !NOTE: just for single machine state :)"""
    logging.info(f"before register bots: {register_ai_room_bots.dict()}")
    is_register = import_bots(bot_name)
    if not is_register:
        logging.info(f"name:{bot_name} not existent bot to import")

    logging.info(f"after register bots: {register_ai_room_bots.dict()}")
    return APIResponse(
        data={
            "is_register": is_register,
            "register_bots": register_ai_room_bots.keys_str(),
        },
    ).model_dump()


# @app.post("/start_bot")
def start_bot(info: RunBotInfo):
    """start run bot"""
    logging.info(f"start bot info:{info}")


"""
curl -XPOST "http://0.0.0.0:4321/bot_join/DummyBot" \
    -H "Content-Type: application/json" \
    -d $'{"config":{"llm":{"messages":[{"role":"system","content":"你是聊天机器人，一个友好、乐于助人的机器人。您的输出将被转换为音频，所以不要包含除“!”以外的特殊字符。'或'?的答案。以一种创造性和有用的方式回应用户 所说的话，但要保持简短。从打招呼开始。"}]},"tts":{"voice":"e90c6678-f0d3-4767-9883-5d0ecf5894a8"}}}' | jq .

curl -XPOST "http://0.0.0.0:4321/bot_join/DailyRTVIBot" \
    -H "Content-Type: application/json" \
    -d $'{"config":{"llm":{"model":"llama-3.1-70b-versatile","messages":[{"role":"system","content":"You are ai assistant. Answer in 1-5 sentences. Be friendly, helpful and concise. Default to metric units when possible. Keep the conversation short and sweet. You only answer in raw text, no markdown format. Don\'t include links or any other extras"}]},"tts":{"voice":"2ee87190-8f84-4925-97da-e52547f9462c"}}}' | jq .

curl -XPOST "http://0.0.0.0:4321/bot_join/DailyAsrRTVIBot" \
    -H "Content-Type: application/json" \
    -d $'{"config":{"llm":{"model":"llama-3.1-70b-versatile","messages":[{"role":"system","content":"你是一位很有帮助中文AI助理机器人。你的目标是用简洁的方式展示你的能力,请用中文简短回答，回答限制在1-5句话内。你的输出将转换为音频，所以不要在你的答案中包含特殊字符。以创造性和有帮助的方式回应用户说的话。"}]},"tts":{"voice":"2ee87190-8f84-4925-97da-e52547f9462c"}}}' | jq .

curl -XPOST "http://0.0.0.0:4321/bot_join/DailyLangchainRAGBot" \
    -H "Content-Type: application/json" \
    -d $'{"config":{"llm":{"model":"llama-3.1-70b-versatile","messages":[{"role":"system","content":""}]},"tts":{"voice":"2ee87190-8f84-4925-97da-e52547f9462c"},"asr":{"tag":"whisper_groq_asr","args":{"language":"en"}}}}' | jq .
"""


@app.post("/realtime_ai/bot_join/{chat_bot_name}")
async def fastapi_bot_join(chat_bot_name: str, info: RunBotInfo) -> JSONResponse:
    # for realtime-ai, no biz code, api design not good
    try:
        res = await bot_join(chat_bot_name, info)
    except Exception as e:
        logging.error(f"Exception in bot_join: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"{e}")
    if res['error_code'] > 0:
        raise HTTPException(status_code=500, detail=f"{res['error_detail']}")
    return JSONResponse({
        "room_url": res['data']['room_url'],
        "token": res['data']['token']
    })


@app.post("/bot_join/{chat_bot_name}")
async def fastapi_bot_join(chat_bot_name: str, info: RunBotInfo) -> JSONResponse:
    # for chat-bot-rtvi-client
    try:
        res = await bot_join(chat_bot_name, info)
    except Exception as e:
        logging.error(f"Exception in bot_join: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"{e}")
    return JSONResponse(res)


async def bot_join(chat_bot_name: str,
                   info: RunBotInfo | dict,
                   services: dict = None,
                   config_list: list = None,
                   config: dict = None) -> dict[str, Any]:
    """join random room chat with bot"""

    logging.info(f"chat_bot_name: {chat_bot_name} request bot info: {info}")
    if isinstance(info, dict):
        info = RunBotInfo(**info)

    if import_bots(chat_bot_name) is False:
        detail = f"un import bot: {chat_bot_name}"
        return APIResponse(error_code=ERROR_CODE_BOT_UN_REGISTER, error_detail=detail).model_dump()

    logging.info(f"register bots: {register_ai_room_bots.items()}")
    if chat_bot_name not in register_ai_room_bots:
        detail = f"bot {chat_bot_name} don't exist"
        return APIResponse(error_code=ERROR_CODE_BOT_UN_REGISTER, error_detail=detail).model_dump()

    exp = RANDOM_ROOM_EXPIRE_TIME
    if chat_bot_name == "DailyLangchainRAGBot":
        exp += DAILYLANGCHAINRAGBOT_EXPIRE_TIME

    daily_room_obj = DailyRoom()
    room = await daily_room_obj.create_room(exp)
    # Give the agent a token to join the session
    bot_token = await daily_room_obj.gen_token(room.name, exp)
    if not room or not bot_token:
        detail = f"Failed to get token for room: {room.name}"
        return APIResponse(error_code=ERROR_CODE_BOT_FAIL_TOKEN, error_detail=detail).model_dump()

    logging.info(f"room: {room}")
    try:
        info.chat_bot_name = chat_bot_name
        info.room_name = room.name
        info.room_url = room.url
        info.token = bot_token
        task_runner = BotTaskRunnerFE(bot_task_mgr, **vars(info))
        await task_runner.run()
    except Exception as e:
        detail = f"bot {chat_bot_name} failed to start process: {e}"
        raise Exception(detail)

    # Grab a token for the user to join with
    user_token = await daily_room_obj.gen_token(room.name, exp)

    data = {
        "room_name": room.name,
        "room_url": room.url,
        "token": user_token,
        "config": task_runner.bot_config,
        "bot_id": task_runner.pid,
        "status": "running",
    }
    return APIResponse(data=data).model_dump()


"""
curl -XPOST "http://0.0.0.0:4321/bot_join/chat-bot/DummyBot" \
    -H "Content-Type: application/json" \
    -d $'{"config":{"llm":{"messages":[{"role":"system","content":"你是聊天机器人，一个友好、乐于助人的机器人。您的输出将被转换为音频，所以不要包含除“!”以外的特殊字符。'或'?的答案。以一种创造性和有用的方式回应用户 所说的话，但要保持简短。从打招呼开始。"}]},"tts":{"voice":"e90c6678-f0d3-4767-9883-5d0ecf5894a8"}}}' | jq .

curl -XPOST "http://0.0.0.0:4321/bot_join/chat-bot/DailyRTVIBot" \
    -H "Content-Type: application/json" \
    -d $'{"config":{"llm":{"model":"llama-3.1-70b-versatile","messages":[{"role":"system","content":"You are ai assistant. Answer in 1-5 sentences. Be friendly, helpful and concise. Default to metric units when possible. Keep the conversation short and sweet. You only answer in raw text, no markdown format. Don\'t include links or any other extras"}]},"tts":{"voice":"2ee87190-8f84-4925-97da-e52547f9462c"}}}' | jq .

curl -XPOST "http://0.0.0.0:4321/bot_join/chat-bot/DailyAsrRTVIBot" \
    -H "Content-Type: application/json" \
    -d $'{"config":{"llm":{"model":"llama-3.1-70b-versatile","messages":[{"role":"system","content":"你是一位很有帮助中文AI助理机器人。你的目标是用简洁的方式展示你的能力,请用中文简短回答，回答限制在1-5句话内。你的输出将转换为音频，所以不要在你的答案中包含特殊字符。以创造性和有帮助的方式回应用户说的话。"}]},"tts":{"voice":"2ee87190-8f84-4925-97da-e52547f9462c"}}}' | jq .

curl -XPOST "http://0.0.0.0:4321/bot_join/chat-bot/DailyLangchainRAGBot" \
    -H "Content-Type: application/json" \
    -d $'{"config":{"llm":{"model":"llama-3.1-70b-versatile","messages":[{"role":"system","content":""}]},"tts":{"voice":"2ee87190-8f84-4925-97da-e52547f9462c"}}}' | jq .
"""


@app.post("/realtime-ai/bot_join/{room_name}/{chat_bot_name}")
async def fastapi_bot_join_room(
        room_name: str,
        chat_bot_name: str,
        info: RunBotInfo) -> JSONResponse:
    try:
        res = await bot_join_room(room_name, chat_bot_name, info)
    except Exception as e:
        logging.error(f"Exception in bot_join_room: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"{e}")

    if res['error_code'] > 0:
        raise HTTPException(status_code=500, detail=f"{res['error_detail']}")
    return JSONResponse({
        "room_url": res['data']['room_url'],
        "token": res['data']['token']
    })


@app.post("/bot_join/{room_name}/{chat_bot_name}")
async def fastapi_bot_join_room(
        room_name: str,
        chat_bot_name: str,
        info: RunBotInfo) -> JSONResponse:
    try:
        res = await bot_join_room(room_name, chat_bot_name, info)
    except Exception as e:
        logging.error(f"Exception in bot_join_room: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"{e}")

    return JSONResponse(res)


async def bot_join_room(room_name: str, chat_bot_name: str, info: RunBotInfo | dict,
                        services: dict = None,
                        config_list: list = None,
                        config: dict = None) -> dict[str, Any]:
    """join room chat with bot"""

    logging.info(f"room_name: {room_name} chat_bot_name: {chat_bot_name} request bot info: {info}")
    if isinstance(info, dict):
        info = RunBotInfo(**info)

    num_bots_in_room = bot_task_mgr.get_bot_proces_num(room_name)
    logging.info(f"num_bots_in_room: {num_bots_in_room}")
    if num_bots_in_room >= MAX_BOTS_PER_ROOM:
        detail = f"Max bot limited reach for room: {room_name}"
        return APIResponse(error_code=ERROR_CODE_BOT_MAX_LIMIT, error_detail=detail).model_dump()

    if import_bots(chat_bot_name) is False:
        detail = f"un import bot: {chat_bot_name}"
        return APIResponse(error_code=ERROR_CODE_BOT_UN_REGISTER, error_detail=detail).model_dump()

    logging.info(f"register bots: {register_ai_room_bots.items()}")
    if chat_bot_name not in register_ai_room_bots:
        detail = f"bot {chat_bot_name} don't exist"
        return APIResponse(error_code=ERROR_CODE_BOT_UN_REGISTER, error_detail=detail).model_dump()

    daily_room_obj = DailyRoom()
    room = await daily_room_obj.create_room(room_name)
    # Give the agent a token to join the session
    bot_token = await daily_room_obj.gen_token(room.name, ROOM_EXPIRE_TIME)
    if not room or not bot_token:
        detail = f"Failed to get token for room: {room.name}"
        return APIResponse(error_code=ERROR_CODE_BOT_FAIL_TOKEN, error_detail=detail).model_dump()

    logging.info(f"room: {room}")
    try:
        info.chat_bot_name = chat_bot_name
        info.room_name = room.name
        info.room_url = room.url
        info.token = bot_token
        task_runner = BotTaskRunnerFE(bot_task_mgr, **vars(info))
        await task_runner.run()
    except Exception as e:
        detail = f"bot {chat_bot_name} failed to start process: {e}"
        raise Exception(detail)

    # Grab a token for the user to join with
    user_token = await daily_room_obj.gen_token(room.name, ROOM_EXPIRE_TIME)

    data = {
        "room_name": room.name,
        "room_url": room.url,
        "token": user_token,
        "config": task_runner.bot_config,
        "bot_id": task_runner.pid,
        "status": "running",
    }
    return APIResponse(data=data).model_dump()


"""
curl -XGET "http://0.0.0.0:4321/status/53187" | jq .
"""


@ app.get("/status/{pid}")
async def fastapi_get_status(pid: int) -> JSONResponse:
    try:
        res = await get_status(pid)
    except Exception as e:
        logging.error(f"Exception in get_status: {e}")
        raise HTTPException(status_code=500, detail=f"{e}")

    return JSONResponse(res)


async def get_status(pid: int) -> dict[str, Any]:
    # Look up the subprocess
    val = bot_task_mgr.get_bot_processor(pid)
    if val is None:
        detail = "Bot not found"
        return APIResponse(error_code=ERROR_CODE_BOT_UN_PROC, error_detail=detail).model_dump()
    proc: multiprocessing.Process = val[0]
    room: GeneralRoomInfo = val[1]
    # If the subprocess doesn't exist, return an error
    if not proc:
        detail = f"Bot with process id: {pid} not found"
        return APIResponse(error_code=ERROR_CODE_BOT_UN_PROC, error_detail=detail).model_dump()

    # Check the status of the subprocess
    if proc.is_alive():
        status = "running"
    else:
        status = "finished"

    data = {
        "bot_id": pid,
        "status": status,
        "room_info": room.model_dump(),
    }
    return APIResponse(data=data).model_dump()


"""
curl -XGET "http://0.0.0.0:4321/room/num_bots/chat-bot" | jq .
"""


@app.get("/room/num_bots/{room_name}")
async def fastapi_get_num_bots(room_name: str):
    try:
        res = await get_num_bots(room_name)
    except Exception as e:
        logging.error(f"Exception in get_num_bots: {e}")
        raise HTTPException(status_code=500, detail=f"{e}")

    return JSONResponse(res)


async def get_num_bots(room_name: str):
    data = {
        "num_bots": bot_task_mgr.get_bot_proces_num(room_name),
    }
    return APIResponse(data=data).model_dump()


"""
curl -XGET "http://0.0.0.0:4321/room/bots/chat-bot" | jq .
"""


@app.get("/room/bots/{room_name}")
async def fastapi_get_room_bots(room_name: str):
    try:
        res = await get_room_bots(room_name)
    except Exception as e:
        logging.error(f"Exception in get_room_bots: {e}")
        raise HTTPException(status_code=500, detail=f"{e}")

    return JSONResponse(res)


async def get_room_bots(room_name: str) -> dict[str, Any]:
    procs = []
    _room = None
    for val in bot_task_mgr.bot_procs.values():
        proc: multiprocessing.Process = val[0]
        room: GeneralRoomInfo = val[1]
        if room.name == room_name:
            procs.append({
                "pid": proc.pid,
                "name": proc.name,
                "status": "running" if proc.is_alive() else "finished",
            })
            _room = room

    if _room is None:
        try:
            daily_room_obj = DailyRoom()
            _room = await daily_room_obj.get_room(room_name)
        except Exception as ex:
            return APIResponse(
                error_code=ERROR_CODE_NO_ROOM,
                error_detail=f"Failed to get room {room_name} from Daily REST API: {ex}",
            ).model_dump()

    response = APIResponse(data={
        "room_info": _room.model_dump(),
        "bots": procs,
    })

    return response.model_dump()


if __name__ == "__main__":
    import uvicorn

    default_host = os.getenv("HOST", "0.0.0.0")
    default_port = int(os.getenv("FAST_API_PORT", "4321"))

    parser = argparse.ArgumentParser(
        description="RTVI Bot Runner")
    parser.add_argument("--host", type=str,
                        default=default_host, help="Host address")
    parser.add_argument("--port", type=int,
                        default=default_port, help="Port number")
    parser.add_argument("--reload", action="store_true",
                        help="Reload code on change")

    config = parser.parse_args()

    # api docs: http://0.0.0.0:4321/docs
    uvicorn.run(
        "src.cmd.http.server.fastapi_daily_bot_serve:app",
        host=config.host,
        port=config.port,
        reload=config.reload
    )
