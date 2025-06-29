# signaling fastapi http service + small webrtc bot (fastapi backgound task)

import os
import argparse
import asyncio
import sys
import logging
from contextlib import asynccontextmanager
import traceback
from typing import Dict

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.cmd.bots.base import AIBot
from src.services.webrtc_peer_connection import SmallWebRTCConnection, IceServer
from src.cmd.bots.bot_loader import BotLoader
from src.common.types import CONFIG_DIR
from src.cmd.bots.base_small_webrtc import SmallWebrtcAIBot


# Load environment variables
load_dotenv(override=True)


run_bot: SmallWebrtcAIBot = None
# Store connections by pc_id
pcs_map: Dict[str, SmallWebRTCConnection] = {}


ice_servers = [
    IceServer(
        urls="stun:stun.l.google.com:19302",
    )
]


# https://fastapi.tiangolo.com/advanced/events/#lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    global run_bot
    try:
        # load model before running
        run_bot = await BotLoader.load_bot(args.f, bot_type="small_webrtc_bot")
    except Exception as e:
        logging.warning(e)
        traceback.print_exc()

    logging.info(f"load chat-bot {run_bot} success")

    yield  # Run app

    logging.info(f"chat-bot app clearing")
    # app life end to clear resources
    coros = [pc.disconnect() for pc in pcs_map.values()]
    await asyncio.gather(*coros)
    pcs_map.clear()

    logging.info(f"chat-bot app clear success")


app = FastAPI(lifespan=lifespan)

# 配置CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境中应该限制为特定域名
    allow_credentials=True,  # 允许携带凭证
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有HTTP头部
)


@app.post("/api/offer")
async def handle_offer(request: dict, background_tasks: BackgroundTasks):
    pc_id = request.get("pc_id")
    logging.info(f"request pc_id: {pc_id}")

    if pc_id and pc_id in pcs_map:
        connection = pcs_map[pc_id]
        logging.info(f"Reusing existing connection for pc_id: {pc_id}")
        await connection.renegotiate(
            sdp=request.get("sdp"),
            type=request.get("type"),
            restart_pc=request.get("restart_pc", False),
        )
    else:
        connection = SmallWebRTCConnection(ice_servers)
        await connection.initialize(
            sdp=request.get("sdp"),
            type=request.get("type"),
        )

        @connection.event_handler("closed")
        async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
            logging.info(f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}")
            pcs_map.pop(webrtc_connection.pc_id, None)

        run_bot.set_webrtc_connection(connection)
        background_tasks.add_task(run_bot.async_run)

    answer = connection.get_answer()
    logging.info(f"answer pc_id: {answer.get('pc_id')}")
    # Updating the peer connection inside the map
    pcs_map[answer.get("pc_id")] = connection

    return answer


"""
python -m src.cmd.webrtc.signaling_bot_server -f config/bots/small_webrtc_server_bot.json 
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC Signaling Bot Server")
    parser.add_argument(
        "--host", default="localhost", help="Host for HTTP server (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=4321, help="Port for HTTP server (default: 4321)"
    )
    parser.add_argument("--reload", action="store_true", help="Reload code on change")
    parser.add_argument(
        "-f",
        type=str,
        default=os.path.join(CONFIG_DIR, "bots/dummy_bot.json"),
        help="Bot configuration json file",
    )

    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
