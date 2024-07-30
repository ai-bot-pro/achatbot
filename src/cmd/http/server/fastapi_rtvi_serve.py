import atexit
import logging
import multiprocessing
import os
import argparse
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

from src.common.logger import Logger
from src.common.interface import IBot
from src.services.help.daily_rest import DailyRESTHelper, \
    DailyRoomObject, DailyRoomProperties, DailyRoomParams
from src.processors.rtvi_processor import RTVIConfig
from src.cmd.bots.base import register_rtvi_bots

from dotenv import load_dotenv
load_dotenv(override=True)

# global logging
Logger.init(logging.DEBUG, is_file=False, is_console=True)


class RoomInfo(BaseModel):
    room_name: str = "chat-bot"
    room_url: str


class AgentInfo(RoomInfo):
    chat_bot_name: str = "DummyBot"
    config: Optional[RTVIConfig] = None


# --------------------- Bot ----------------------------

MAX_BOTS_PER_ROOM = 10

# Bot sub-process dict for status reporting and concurrency control
bot_procs = {}


def cleanup():
    # Clean up function, just to be extra safe
    for pid, (proc, url) in bot_procs.items():
        if proc.is_alive():
            proc.join()
            proc.close()
            logging.info(f"pid:{pid} url:{url} proc: {proc} close")
        else:
            logging.warning(f"pid:{pid} url:{url} proc: {proc} already closed")


atexit.register(cleanup)

# ------------ Fast API Config ------------ #

DAILY_API_URL = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")
DAILY_API_KEY = os.getenv("DAILY_API_KEY", "")
ROOM_TOKEN_EXPIRE_TIME = 30 * 60  # 30 minutes

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


@app.get("/create_room/{name}")
def create_room(name):
    # Create a Daily rest helper
    daily_rest_helper = DailyRESTHelper(DAILY_API_KEY, DAILY_API_URL)
    # Create a new room
    try:
        params = DailyRoomParams(name=name, properties=DailyRoomProperties())
        room: DailyRoomObject = daily_rest_helper.create_room(params=params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}")

    # Give the agent a token to join the session
    token = daily_rest_helper.get_token(room.url, ROOM_TOKEN_EXPIRE_TIME)

    if not room or not token:
        raise HTTPException(
            status_code=500, detail=f"Failed to get token for room: {room.name}")

    return RedirectResponse(room.url)


@app.post("/agent_join/{bot_name}")
async def agent_join(bot_name: str, request: Request) -> JSONResponse:
    logging.debug(f"register bots: {register_rtvi_bots.items()}")
    if bot_name not in register_rtvi_bots:
        raise HTTPException(status_code=500, detail=f"bot {bot_name} don't exist")

    try:
        data = await request.json()
        if "config" not in data:
            raise Exception("Missing RTVI configuration object for bot")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}")

    try:
        logging.debug(f'config: {data["config"]}')
        bot_config = RTVIConfig(**data["config"])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Failed to parse bot configuration")
    # Create a Daily rest helper
    daily_rest_helper = DailyRESTHelper(DAILY_API_KEY, DAILY_API_URL)

    room: DailyRoomObject | None = None
    try:
        room = daily_rest_helper.get_room_from_name(bot_name)
    except Exception as ex:
        logging.warning(f"Failed to get room {bot_name} from Daily REST API: {ex}")
        # Create a new room
        try:
            params = DailyRoomParams(name=bot_name, properties=DailyRoomProperties())
            room = daily_rest_helper.create_room(params=params)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")

    # Give the agent a token to join the session
    token = daily_rest_helper.get_token(room.url, ROOM_TOKEN_EXPIRE_TIME)

    if not room or not token:
        raise HTTPException(
            status_code=500, detail=f"Failed to get token for room: {room.name}")

    logging.info(f"room: {room}")
    pid = 0
    try:
        bot_obj: IBot = register_rtvi_bots[bot_name](room.url, token, bot_config, bot_name)
        bot_process: multiprocessing.Process = multiprocessing.Process(target=bot_obj.run)
        bot_process.start()
        pid = bot_process.pid
        bot_procs[pid] = (bot_process, room.url)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"bot {bot_name} failed to start process: {e}")

    # Grab a token for the user to join with
    user_token = daily_rest_helper.get_token(room.url, ROOM_TOKEN_EXPIRE_TIME)

    return JSONResponse({
        "room_name": room.name,
        "room_url": room.url,
        "token": user_token,
        "bot_config": bot_config.model_dump_json(),
        "bot_id": pid,
        "status": "running",
    })


@app.get("/status/{pid}")
def get_status(pid: int):
    # Look up the subprocess
    proc, url = bot_procs.get(pid)

    # If the subprocess doesn't exist, return an error
    if not proc:
        raise HTTPException(
            status_code=404, detail=f"Bot with process id: {pid} not found")

    # Check the status of the subprocess
    if proc.is_alive():
        status = "running"
    else:
        status = "finished"

    return JSONResponse({"bot_id": pid, "status": status, "room_url": url})


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

    uvicorn.run(
        "src.cmd.http.server.fastapi_rtvi_serve:app",
        host=config.host,
        port=config.port,
        reload=config.reload
    )
