import asyncio
import logging
import os
import argparse
from contextlib import asynccontextmanager
import traceback
from typing import Dict, List
import json
from asyncio import TimeoutError


from fastapi import FastAPI, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi import WebSocketDisconnect
from dotenv import load_dotenv
from fastapi.websockets import WebSocketState

from src.common.utils.helper import ThreadSafeDict
from src.cmd.bots.bridge.base import AISmallWebRTCFastapiWebsocketBot
from src.cmd.bots.base import AIBot
from src.cmd.bots.bot_loader import BotLoader
from src.common.types import CONFIG_DIR
from src.common.const import *
from src.common.logger import Logger
from src.cmd.http.server.fastapi_daily_bot_serve import ngrok_proxy
from src.services.webrtc_peer_connection import SmallWebRTCConnection, IceServer


load_dotenv(override=True)
Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)

run_bot: AISmallWebRTCFastapiWebsocketBot = None
config = None

# TODO: connect session mrg
# Store websocket connection
ws_map = ThreadSafeDict()
# Store rtc connections
pcs_map = ThreadSafeDict()
# Store peer rtc connection pending candidates
pending_candidates = ThreadSafeDict()

ice_servers = [
    IceServer(
        urls="stun:stun.l.google.com:19302",
    )
]


# https://fastapi.tiangolo.com/advanced/events/#lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    global run_bot, ws_map, pcs_map, pending_candidates
    try:
        # load model before running
        config_file = config.f if config else os.getenv("CONFIG_FILE")
        run_bot = await BotLoader.load_bot(config_file, bot_type="fastapi_ws_bot")
        run_bot.load()
    except Exception as e:
        print(e)
        traceback.print_exc()
        return

    print(f"load bot {run_bot} success")

    yield  # Run app

    # app life end to clear resources
    # clear websocket connection
    coros = [ws.close() for ws in ws_map.values() if ws.client_state == WebSocketState.CONNECTED]
    await asyncio.gather(*coros)
    # Create new empty ThreadSafeDicts instead of clearing
    ws_map = ThreadSafeDict()
    print(f"websocket connections clear success")

    # clear webrtc connection
    coros = [pc.disconnect() for pc in pcs_map.values() if pc.connectionState == "connected"]
    await asyncio.gather(*coros)
    pcs_map = ThreadSafeDict()
    pending_candidates = ThreadSafeDict()
    print(f"rtc peer connections clear success")


app = FastAPI(lifespan=lifespan)

# 配置CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境中应该限制为特定域名
    allow_credentials=True,  # 允许携带凭证
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有HTTP头部
)


async def handle_ice_candidate(candidate: dict, peer_id: str):
    """
    post remote peer ice candidate
    """
    if not candidate:
        return

    logging.info(f"received ice candidate from {peer_id}: {candidate.get('candidate_sdp')}...")

    if not pending_candidates.get(peer_id):
        pending_candidates.set(peer_id, [])

    if not pcs_map.get(peer_id):
        pending_candidates_list: List[dict] = pending_candidates.get(peer_id)
        pending_candidates_list.append(candidate)
        pending_candidates.set(peer_id, pending_candidates_list)
    else:
        pending_candidates_list: List[dict] = pending_candidates.get(peer_id)
        connection: SmallWebRTCConnection = pcs_map.get(peer_id)
        if pending_candidates_list and len(pending_candidates_list) > 0:
            await asyncio.gather(
                *(connection.add_ice_candidate(c) for c in pending_candidates_list)
            )
            pending_candidates.set(peer_id, [])
        await connection.add_ice_candidate(candidate)


async def handle_offer(request: dict, peer_id: str):
    logging.info(f"request: {request} peer_id: {peer_id}")

    if not peer_id:
        logging.error(f"peer_id is empty")
        return None

    if peer_id and pcs_map.contains(peer_id):
        connection: SmallWebRTCConnection = pcs_map.get(peer_id)
        logging.info(f"Reusing existing connection for peer_id: {peer_id}")
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
            logging.info(f"Discarding peer connection for peer_id: {peer_id}")

            # Remove from pcs_map
            pcs_map.pop(peer_id)

            # Close associated websocket if it exists
            if ws_map.contains(peer_id):
                try:
                    ws: WebSocket = ws_map.get(peer_id)
                    if ws.client_state == WebSocketState.CONNECTED:
                        await ws.close(code=1000, reason="WebRTC connection closed")
                    ws_map.pop(peer_id)
                    logging.info(
                        f"Closed websocket for peer_id: {peer_id} due to WebRTC disconnection"
                    )
                except Exception as e:
                    logging.error(f"Error closing websocket for {peer_id}: {str(e)}")

    answer = connection.get_answer()
    logging.info(f"answer bot peer_id: {answer.get('pc_id')}")
    pcs_map.set(peer_id, connection)

    return answer


# 添加心跳超时设置
HEARTBEAT_TIMEOUT = 35  # 35 seconds (slightly longer than client's 30-second interval)


@app.websocket("/{peer_id}")
async def websocket_endpoint(websocket: WebSocket, peer_id: str):
    try:
        await websocket.accept()
        ws_map.set(peer_id, websocket)
        logging.info(f"accept {peer_id=} client: {websocket.client}")

        # Start the heartbeat monitoring loop
        last_heartbeat = asyncio.get_event_loop().time()
        bot_task = None

        try:
            while True:
                try:
                    # Wait for a message with timeout
                    message = await asyncio.wait_for(
                        websocket.receive_text(), timeout=HEARTBEAT_TIMEOUT
                    )

                    try:
                        data = json.loads(message)
                        message_type = data.get("type")

                        if message_type == "ping":
                            # Respond to heartbeat
                            await websocket.send_text(json.dumps({"type": "pong"}))
                            last_heartbeat = asyncio.get_event_loop().time()
                            continue

                        # 处理offer消息
                        elif message_type == "offer":
                            logging.info(
                                f"Received offer message via WebSocket from peer: {peer_id}"
                            )

                            # 创建后台任务对象
                            # background_tasks = BackgroundTasks()

                            # 处理offer请求
                            answer = await handle_offer(data, peer_id)

                            if answer:
                                # 通过WebSocket发送answer响应
                                await websocket.send_text(
                                    json.dumps(
                                        {
                                            "type": "answer",
                                            "sdp": answer.get("sdp"),
                                            "type_value": answer.get("type"),
                                        }
                                    )
                                )

                                # 如果这是第一次连接，设置WebRTC连接并启动bot
                                if pcs_map.contains(peer_id) and not bot_task:
                                    run_bot.set_webrtc_connection(pcs_map.get(peer_id))
                                    run_bot.set_fastapi_websocket(websocket)
                                    bot_task = asyncio.create_task(run_bot.async_run())
                            else:
                                logging.error(f"Failed to create answer for peer: {peer_id}")

                        # 处理ice_candidate消息
                        elif message_type == "ice_candidate":
                            logging.info(
                                f"Received ICE candidate via WebSocket from peer: {peer_id}"
                            )

                            # 提取ICE候选信息
                            candidate = {
                                "candidate_sdp": data.get("candidate_sdp"),
                                "sdpMid": data.get("sdpMid"),
                                "sdpMLineIndex": data.get("sdpMLineIndex"),
                                "usernameFragment": data.get("usernameFragment"),
                            }

                            # 处理ICE候选
                            await handle_ice_candidate(candidate, peer_id)

                            # 发送响应确认
                            await websocket.send_text(
                                json.dumps({"type": "ice_candidate_response", "status": "success"})
                            )

                        else:
                            # 其他类型的消息，记录但不处理
                            logging.info(
                                f"Received unknown message type: {message_type} from peer: {peer_id}"
                            )

                    except json.JSONDecodeError:
                        # Not a JSON message, ignore
                        pass

                except TimeoutError:
                    # Check if we've exceeded the heartbeat timeout
                    if asyncio.get_event_loop().time() - last_heartbeat > HEARTBEAT_TIMEOUT:
                        logging.warning(f"Heartbeat timeout for peer {peer_id}")
                        break
                    continue
                except WebSocketDisconnect:
                    logging.info(f"Client {peer_id} disconnected")
                    break

        finally:
            # Clean up
            if ws_map.contains(peer_id):
                ws_map.delete(peer_id)
            if bot_task and not bot_task.done():
                bot_task.cancel()
                try:
                    await bot_task
                    bot_task = None
                except asyncio.CancelledError:
                    pass
            logging.info(f"WebSocket connection closed for peer {peer_id}")

    except Exception as e:
        logging.error(f"Error in websocket endpoint: {str(e)}", exc_info=True)
        if ws_map.contains(peer_id):
            ws_map.delete(peer_id)


"""
python -m src.cmd.webrtc_websocket.fastapi_ws_signaling_bot_serve_v2 -f config/bots/small_webrtc_fastapi_websocket_avatar_echo_bot.json
"""

if __name__ == "__main__":
    import uvicorn

    default_host = os.getenv("HOST", "0.0.0.0")
    default_port = int(os.getenv("FAST_API_PORT", "4321"))

    parser = argparse.ArgumentParser(description="Fastapi Websocket Bot Runner")
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

    config = parser.parse_args()

    if config.ngrok:
        ngrok_proxy(config.port)

    # Note: not event loop to new on    # api docs: http://0.0.0.0:4321/docs
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        reload=config.reload,
    )
