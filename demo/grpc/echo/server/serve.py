import grpc
from concurrent import futures
import logging


from demo.grpc.idl.echo_pb2_grpc import add_EchoServicer_to_server
from demo.grpc.echo.server.servicers.echo import Echo
from demo.grpc.interceptors.authentication_server import AuthenticationInterceptor


logging.basicConfig(level=logging.DEBUG)


def serve() -> None:
    logging.basicConfig()
    token = "oligei"
    authenticator = AuthenticationInterceptor(
        "authorization", token, grpc.StatusCode.UNAUTHENTICATED, "Access denied!"
    )
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        interceptors=(authenticator,),
    )
    add_EchoServicer_to_server(Echo(), server)
    port = 50051
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print("Server started")
    server.wait_for_termination()


# python -m demo.grpc.echo.server.serve 2> err_serve.log
if __name__ == "__main__":
    serve()
