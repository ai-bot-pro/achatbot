import asyncio
import logging
import os
import argparse
from contextlib import asynccontextmanager
import traceback
from typing import Dict

from fastapi import FastAPI, WebSocket
from dotenv import load_dotenv

from src.cmd.bots.base import AIBot
from src.cmd.bots.bot_loader import BotLoader
from src.cmd.bots.base_fastapi_websocket_server import AIFastapiWebsocketBot
from src.common.types import CONFIG_DIR
from src.common.const import *
from src.common.logger import Logger
from src.cmd.http.server.fastapi_daily_bot_serve import ngrok_proxy


load_dotenv(override=True)
Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)

run_bot: AIFastapiWebsocketBot = None
# Store websocket
ws_map: Dict[str, WebSocket] = {}


# https://fastapi.tiangolo.com/advanced/events/#lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    global run_bot
    try:
        # load model before running
        run_bot = await BotLoader.load_bot(config.f, bot_type="fastapi_ws_bot")
    except Exception as e:
        print(e)
        traceback.print_exc()

    print(f"load bot {run_bot} success")

    yield  # Run app

    # clear
    coros = [ws.close() for ws in ws_map.values() if ws.state == "OPEN"]
    await asyncio.gather(*coros)
    ws_map.clear()
    print(f"clear success")


app = FastAPI(lifespan=lifespan)


@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    # NOTE: after init, websocket to accept connection, then to run
    await websocket.accept()
    key = f"{websocket.client.host}:{websocket.client.port}"
    ws_map[key] = websocket
    run_bot.set_fastapi_websocket(websocket)
    logging.info(f"accept client: {websocket.client}")
    await run_bot.async_run()


"""
python -m src.cmd.websocket.server.fastapi_ws_bot_serve -f config/bots/fastapi_websocket_echo_voice_bot.json

python -m src.cmd.websocket.server.fastapi_ws_bot_serve -f config/bots/fastapi_websocket_server_bot.json
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

    # Note: not event loop to new one
    # run_bot: AIFastapiWebsocketBot = asyncio.get_event_loop().run_until_complete(BotLoader.load_bot(config.f, bot_type="fastapi_ws_bot"))

    # use one event loop to run
    # run_bot: AIFastapiWebsocketBot = asyncio.run(
    #    BotLoader.load_bot(config.f, bot_type="fastapi_ws_bot"))

    # api docs: http://0.0.0.0:4321/docs
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        reload=config.reload,
    )
