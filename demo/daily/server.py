"""
- https://docs.daily.co/guides/products/ai-toolkit
- https://reference-python.daily.co/api_reference.html
- https://docs.daily.co/guides

use daily-python (wrap rust client call lib.so)

- more details: https://docs.cerebrium.ai/v4/examples/realtime-voice-agents
"""

import subprocess
import logging
import atexit
import os
import typing

import argparse
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

from . import daily_helpers

MAX_BOTS_PER_ROOM = 10

# Bot sub-process dict for status reporting and concurrency control
bot_procs = {}


class RoomInfo(BaseModel):
    room_name: str = "chat-bot"
    room_url: str


class AgentInfo(RoomInfo):
    is_agent: bool = False
    chat_bot_name: str = "chat_bot_pyaudio_echo"
    args: str = "-e 1"


def cleanup():
    # Clean up function, just to be extra safe
    for proc, url in bot_procs.values():
        proc.terminate()
        proc.wait()
        print(f"url:{url} proc: {proc} terminate")


atexit.register(cleanup)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/create_room/{name}")
def create_room(name):
    try:
        room_url, room_name = daily_helpers.create_room(name)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to create room: {ex}")
    print(f"create ok, room_url:{room_url},room_name:{room_name}")
    # Ensure the room property is present
    if not room_url:
        raise HTTPException(
            status_code=500,
            detail="Missing 'room' property in request data. Cannot start agent without a target room!",
        )

    return RedirectResponse(room_url)


"""
curl -XPOST "http://0.0.0.0:6789/agent_join/" \
    -H "Content-Type: application/json" \
    -d '{"room_url": "https://weedge.daily.co/chat-bot","room_name":"chat-bot"}'

curl -XPOST "http://0.0.0.0:6789/agent_join/" \
    -H "Content-Type: application/json" \
    -d '{"room_url": "https://weedge.daily.co/chat-bot","room_name":"chat-bot","chat_bot_name":"chat_bot_daily_echo","args":"--receive 1 --send 1"}'
"""


@app.post("/token/")
def token(info: AgentInfo):
    # Get the token for the room
    token = daily_helpers.get_token(info.room_url)
    return JSONResponse({"token": token})


@app.post("/agent_join/")
def agent_join(info: AgentInfo):
    room_url = info.room_url
    # Check if there is already an existing process running in this room
    num_bots_in_room = sum(
        1 for proc in bot_procs.values() if proc[1] == room_url and proc[0].poll() is None
    )
    if num_bots_in_room >= MAX_BOTS_PER_ROOM:
        raise HTTPException(status_code=500, detail=f"Max bot limited reach for room: {room_url}")

    # Get the token for the room
    token = daily_helpers.get_token(room_url)
    if not token:
        raise HTTPException(status_code=500, detail=f"Failed to get token for room: {room_url}")

    # Spawn a new agent, and join the user session
    # Note: this is mostly for demonstration purposes
    cmd = f"cd ../.. && python3 -m demo.daily.{info.chat_bot_name} -u {room_url} -t {token} {info.args}"
    if info.is_agent:
        cmd = f"cd ../.. && python3 -m demo.daily.agents.{info.chat_bot_name} -u {room_url} -t {token} {info.args}"
    print(cmd)
    try:
        proc = subprocess.Popen(
            [cmd], shell=True, bufsize=1, cwd=os.path.dirname(os.path.abspath(__file__))
        )
        bot_procs[proc.pid] = (proc, room_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start subprocess: {e}")

    return JSONResponse({"bot_id": proc.pid, "status": "running"})


@app.get("/status/{pid}")
def get_status(pid: int):
    # Look up the subprocess
    proc, url = bot_procs.get(pid)

    # If the subprocess doesn't exist, return an error
    if not proc:
        raise HTTPException(status_code=404, detail=f"Bot with process id: {pid} not found")

    # Check the status of the subprocess
    if proc[0].poll() is None:
        status = "running"
    else:
        status = "finished"

    return JSONResponse({"bot_id": pid, "status": status, "room_url": url})


if __name__ == "__main__":
    import uvicorn

    default_host = os.getenv("HOST", "0.0.0.0")
    default_port = int(os.getenv("FAST_API_PORT", "6789"))

    parser = argparse.ArgumentParser(description="Daily Storyteller FastAPI server")
    parser.add_argument("--host", type=str, default=default_host, help="Host address")
    parser.add_argument("--port", type=int, default=default_port, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Reload code on change")

    config = parser.parse_args()

    uvicorn.run(
        "demo.daily.server:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
    )
