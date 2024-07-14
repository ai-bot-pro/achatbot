import grpc
from concurrent import futures
import logging
import os


from src.common.logger import Logger
from src.cmd.grpc.idl.tts_pb2_grpc import add_TTSServicer_to_server
from src.cmd.grpc.interceptors.authentication_server import AuthenticationInterceptor
from src.cmd.grpc.speaker.server.servicers.tts import TTS

Logger.init(logging.DEBUG, app_name="chat-bot-tts-serve", is_file=True, is_console=True)


def serve() -> None:
    port = os.getenv('PORT', "50052")
    max_workers = int(os.getenv('MAX_WORKERS', "10"))
    logging.info(f"serve port: {port} max_workers: {max_workers}")
    token = "oligei-tts"
    authenticator = AuthenticationInterceptor(
        'authorization', token,
        grpc.StatusCode.UNAUTHENTICATED, 'Access denied!'
    )
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        interceptors=(authenticator,),
    )
    add_TTSServicer_to_server(TTS(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logging.info(f"Server started port: {port}")
    server.wait_for_termination()


# python -m src.cmd.grpc.speaker.server.serve
if __name__ == "__main__":
    serve()
