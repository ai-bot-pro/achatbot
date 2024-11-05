import json
import logging
import os
import argparse

from fastapi import FastAPI, WebSocket

from src.cmd.bots.fastapi_websocket_server_bot import FastapiWebsocketServerBot
from src.common.types import CONFIG_DIR, BotRunArgs
from src.cmd.bots.run import RunBotInfo
from src.common.const import *
from src.common.logger import Logger
from src.cmd.http.server.fastapi_daily_bot_serve import app, ngrok_proxy


from dotenv import load_dotenv
load_dotenv(override=True)


Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)

"""
!TODO: load bot config from remote or use custom config file
"""


async def load_bot_config_from_local(file_path: str):
    bot_config = {}
    with open(file_path, 'r') as f:
        bot_config = json.load(f)
        print(json.dumps(bot_config, indent=4, sort_keys=True))
    bot_info = RunBotInfo(**bot_config)
    logging.info(f"bot_info:{bot_info}")
    return bot_info


@app.websocket("/{chat_bot_name}")
async def websocket_endpoint(websocket: WebSocket, chat_bot_name: str):
    logging.info(f"bot:{chat_bot_name} client: {websocket.client}")

    local_file_path = os.path.join(CONFIG_DIR, "bots", f"{chat_bot_name}.json")
    print(local_file_path)
    if not os.path.isfile(local_file_path):
        return
    bot_info = await load_bot_config_from_local(file_path=local_file_path)

    await websocket.accept()
    logging.info(f"bot:{chat_bot_name} accept client: {websocket.client}")

    bot_args = BotRunArgs(
        bot_name=chat_bot_name,
        bot_config=bot_info.config,
        bot_config_list=bot_info.config_list,
        services=bot_info.services,
    )
    run_bot = FastapiWebsocketServerBot(websocket=websocket, **vars(bot_args))
    await run_bot.arun()


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
    parser.add_argument("--ngrok", type=bool,
                        default=False, help="use ngrok proxy")

    config = parser.parse_args()

    if config.ngrok:
        ngrok_proxy(config.port)

    # api docs: http://0.0.0.0:4321/docs
    uvicorn.run(
        "src.cmd.http.server.fastapi_daily_bot_serve:app",
        host=config.host,
        port=config.port,
        reload=config.reload
    )
