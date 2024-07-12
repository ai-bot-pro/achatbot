import grpc
from concurrent import futures
import logging


from src.cmd.grpc.idl.tts_pb2_grpc import add_TTSServicer_to_server
from src.cmd.grpc.interceptors.authentication_server import AuthenticationInterceptor
from src.cmd.grpc.speaker.server.servicers.tts import TTS

logging.basicConfig(level=logging.DEBUG)


def serve() -> None:
    logging.basicConfig()
    token = "oligei-tts"
    authenticator = AuthenticationInterceptor(
        'authorization', token,
        grpc.StatusCode.UNAUTHENTICATED, 'Access denied!'
    )
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        interceptors=(authenticator,),
    )
    add_TTSServicer_to_server(TTS(), server)
    port = 50052
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print("Server started")
    server.wait_for_termination()


# python -m src.cmd.grpc.speaker.server.serve 2> .log/tts_serve_err.log
if __name__ == "__main__":
    serve()
