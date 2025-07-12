"""
@author: weege007@gmail.com

run webrtc channel/room bot worker with config
- main processor to init bot to load
- use fastapi http background to run

@TODO: use queue connenctor(e.g.: zmq, redis) to run webrtc channel/room bot worker with config
"""

import logging
from multiprocessing import current_process
import os
import argparse
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi.responses import JSONResponse

from src.common.types import CONFIG_DIR, GeneralRoomInfo
from src.common.logger import Logger
from src.common.const import *
from src.cmd.bots.base import AIBot
from src.cmd.bots.bot_loader import BotLoader, load_bot_config_from_local
from src.cmd.bots.run import RunBotInfo
from src.cmd.bots import import_bots, register_ai_room_bots
from src.cmd.http.server.help import (
    ERROR_CODE_BOT_UN_REGISTER,
    ERROR_CODE_VALID_ROOM,
    APIResponse,
    check_host_whitelist,
    ngrok_proxy,
    getRoomMgr,
)


load_dotenv(override=True)
Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)

run_bot: AIBot = None
bot_info: RunBotInfo = None
cmd_args = None


async def create_room(bot_info: RunBotInfo):
    room_obj = getRoomMgr(bot_info)
    is_valid_room = await room_obj.check_valid_room(bot_info.room_name, bot_info.token)
    room: GeneralRoomInfo = None
    if is_valid_room is False:
        room = await room_obj.create_room(bot_info.room_name)
    else:
        room = await room_obj.get_room(bot_info.room_name)

    # Give the agent a token to join the session
    bot_token = await room_obj.gen_token(room.name, bot_info.room_expire)
    if not room or not bot_token:
        detail = f"Failed to get token for room: {room.name}"
        raise Exception(detail)

    logging.info(f"room: {room} bot_token:{bot_token}")

    return room, bot_token


# https://fastapi.tiangolo.com/advanced/events/#lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    global run_bot, bot_info, cmd_args
    try:
        cur_process = current_process()
        assert cur_process.name == "MainProcess"

        # load bot before running
        config_file = cmd_args.f if cmd_args else os.getenv("CONFIG_FILE")
        bot_info = load_bot_config_from_local(config_file)
        room, bot_token = await create_room(bot_info)
        run_bot = await BotLoader.load_bot(config_file, bot_type="room_bot")
        run_bot.set_args(
            {
                "room_name": room.name,
                "room_url": room.url,
                "token": bot_token,
            }
        )
        run_bot.load()
        logging.info(f"load bot success,{run_bot.args=}")
    except Exception as e:
        logging.error(e, exc_info=True)
        return

    yield  # Run app

    # app life end to clear resources


app = FastAPI(lifespan=lifespan)

# 配置CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境中应该限制为特定域名
    allow_credentials=True,  # 允许携带凭证
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有HTTP头部
)

# ------------ Fast API Routes ------------ #
# https://fastapi.tiangolo.com/async/


@app.middleware("http")
async def allowed_hosts_middleware(request: Request, call_next):
    # Middle that optionally checks for hosts in a whitelist
    if not check_host_whitelist(request):
        raise HTTPException(status_code=403, detail="Host access denied")
    response = await call_next(request)
    return response


@app.get("/health")
async def health():
    return JSONResponse(APIResponse().model_dump())


@app.get("/readiness")
async def readiness():
    # todo
    return JSONResponse(APIResponse().model_dump())


"""
curl -XPOST "http://0.0.0.0:4321/bot_join/chat-room/DummyBot" | jq .
"""


@app.post("/bot_join/{room_name}/{chat_bot_name}")
async def fastapi_bot_join_room(
    room_name: str,
    chat_bot_name: str,
    request: Request,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    try:
        token_id = request.get("token_id", None)
        res = await bot_join_room(room_name, chat_bot_name, background_tasks, token_id)
    except Exception as e:
        logging.error(f"Exception in bot_join_room: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"{e}")

    return JSONResponse(res)


async def bot_join_room(
    room_name: str,
    chat_bot_name: str,
    background_tasks: BackgroundTasks,
    token_id: str = None,
) -> dict[str, Any]:
    """join room chat with bot config"""
    logging.info(f"room_name: {room_name} chat_bot_name: {chat_bot_name}")
    if import_bots(chat_bot_name) is False:
        detail = f"un import bot: {chat_bot_name}"
        return APIResponse(error_code=ERROR_CODE_BOT_UN_REGISTER, error_detail=detail).model_dump()

    logging.info(f"register bots: {register_ai_room_bots.items()}")
    if chat_bot_name not in register_ai_room_bots:
        detail = f"bot {chat_bot_name} don't exist"
        return APIResponse(error_code=ERROR_CODE_BOT_UN_REGISTER, error_detail=detail).model_dump()

    token_id = token_id or bot_info.token
    room_obj = getRoomMgr(bot_info)
    is_valid_room = await room_obj.check_valid_room(room_name, bot_info.token)
    if is_valid_room is False:
        detail = f"not valid room: {room_name}"
        return APIResponse(error_code=ERROR_CODE_VALID_ROOM, error_detail=detail).model_dump()

    background_tasks.add_task(run_bot.async_run)

    # Grab a token for the user/agent to join with
    user_token = await room_obj.gen_token(room_name, bot_info.room_expire)

    data = {
        "room_name": run_bot.args.room_name,
        "room_url": run_bot.args.room_url,
        "token": user_token,
        "room_expire": bot_info.room_expire,
        "config": run_bot.bot_config(),
        "bot_id": os.getppid(),
        "bot_name": run_bot.args.bot_name,
        "status": "running",
    }
    return APIResponse(data=data).model_dump()


"""
python -m src.cmd.http.server.fastapi_room_bot_serve -f config/bots/dummy_bot.json
"""

if __name__ == "__main__":
    import uvicorn

    default_host = os.getenv("HOST", "0.0.0.0")
    default_port = int(os.getenv("FAST_API_PORT", "4321"))

    parser = argparse.ArgumentParser(description="Fastapi http Room Bot Runner")
    parser.add_argument("--host", type=str, default=default_host, help="Host address")
    parser.add_argument("--port", type=int, default=default_port, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Reload code on change")
    parser.add_argument("--ngrok", action="store_true", help="use ngrok proxy")
    parser.add_argument(
        "-f",
        type=str,
        default=os.path.join(CONFIG_DIR, "bots/dummy_bot.json"),
        help="Bot configuration json file",
    )

    cmd_args = parser.parse_args()

    if cmd_args.ngrok:
        ngrok_proxy(cmd_args.port)

    # Note: not event loop to new on    # api docs: http://0.0.0.0:4321/docs
    uvicorn.run(
        app,
        host=cmd_args.host,
        port=cmd_args.port,
        reload=cmd_args.reload,
    )
