from concurrent import futures
import logging
import os
import sys
import threading
import queue

import grpc
from dotenv import load_dotenv

try:
    cur_dir = os.path.dirname(__file__)
    sys.path.insert(1, os.path.join(cur_dir, "../../../../common/grpc/idl"))
    from src.common.grpc.idl.tts_pb2_grpc import add_TTSServicer_to_server
except ModuleNotFoundError as e:
    raise Exception(f"grpc import error: {e}")

from src.common.grpc.interceptors.authentication_server import AuthenticationInterceptor
from src.cmd.grpc.speaker.server.servicers.tts import TTS, main_thread_init
from src.common.logger import Logger

load_dotenv(override=True)

Logger.init(
    os.getenv("LOG_LEVEL", "debug").upper(),
    app_name="chat-bot-tts-serve",
    is_file=False,
    is_console=True,
)


def start_grpc_server(tts_service: TTS):
    # start grpc server
    port = os.getenv("PORT", "50052")
    max_workers = int(os.getenv("MAX_WORKERS", "10"))
    logging.info(f"serve port: {port} max_workers: {max_workers}")
    token = "oligei-tts"
    authenticator = AuthenticationInterceptor(
        "authorization", token, grpc.StatusCode.UNAUTHENTICATED, "Access denied!"
    )
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        interceptors=(authenticator,),
    )
    add_TTSServicer_to_server(tts_service, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logging.info(f"Server started port: {port}")
    server.wait_for_termination()


def serve() -> None:
    # run tts init
    tts_service = TTS()

    grpc_thread = threading.Thread(
        target=start_grpc_server,
        args=(tts_service,),
        daemon=True,
    )
    grpc_thread.start()

    main_thread_init(tts_service)


# python -m src.cmd.grpc.speaker.server.serve
if __name__ == "__main__":
    serve()
