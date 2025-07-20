import os
from typing import Any
from fastapi import Request
from pydantic import BaseModel

from src.cmd.bots.run import RunBotInfo
from src.common.interface import IRoomManager
from src.services.help import RoomManagerEnvInit


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
ERROR_CODE_VALID_ROOM = 10005

MAX_BOTS_PER_ROOM = 10

# ------------ Helper methods ------------ #


def ngrok_proxy(port):
    from pyngrok import ngrok
    import nest_asyncio

    ngrok_tunnel = ngrok.connect(port)
    print("Public URL:", ngrok_tunnel.public_url)
    nest_asyncio.apply()


def getRoomMgr(run_bot_info: RunBotInfo = None) -> IRoomManager:
    room_mgr: IRoomManager = None
    if run_bot_info and run_bot_info.room_manager:
        room_mgr = RoomManagerEnvInit.initEngine(
            run_bot_info.room_manager.tag, run_bot_info.room_manager.args
        )
    else:
        room_mgr = RoomManagerEnvInit.initEngine()
    return room_mgr


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
