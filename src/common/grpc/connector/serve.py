from concurrent import futures
import threading
import logging
import queue
import os

import grpc

from src.common.grpc.idl.connector_pb2_grpc import (
    ConnectorServicer, add_ConnectorServicer_to_server
)
from src.common.grpc.idl.connector_pb2 import (
    ConnectStreamResponse
)
from src.common.grpc.interceptors.authentication_server import AuthenticationInterceptor
from src.common.logger import Logger


class Connector(ConnectorServicer):
    def __init__(self, in_q: queue.Queue, out_q: queue.Queue) -> None:
        super().__init__()
        self.in_q = in_q
        self.out_q = out_q
        self.is_active = False
        self.recv_thread = None

    def ConnectStream(self, request_iterator, context):
        def recv_to_queue(request_iterator):
            for request in request_iterator:
                self.in_q.put(request)

        if self.recv_thread is None or self.recv_thread.is_alive() is False:
            self.recv_thread = threading.Thread(
                target=recv_to_queue, args=(request_iterator,))
            self.recv_thread.start()
            self.is_active = True

            while self.is_active:
                frame = self.out_q.get()
                yield ConnectStreamResponse(frame=frame)

    def close(self):
        self.is_active = False
        if self.recv_thread is not None and self.recv_thread.is_alive():
            self.recv_thread.join()


class StreamServe():
    def __init__(self, in_q: queue.Queue, out_q: queue.Queue,
                 port="50052", max_workers=1) -> None:
        self.in_q = in_q
        self.out_q = out_q
        self.port = port
        self.max_workers = max_workers
        self.serve_thread: threading.Thread = None
        self.connector = Connector(in_q, out_q)

    def start(self):
        if self.serve_thread is None or self.serve_thread.is_alive() is False:
            self.serve_thread = threading.Thread(target=self._serve, args=())
            self.serve_thread.start()

    def stop(self):
        if self.serve_thread is not None or self.serve_thread.is_alive():
            self.serve_thread.join()

    def close(self):
        self.connector.close()
        self.stop()

    def _serve(self) -> None:
        logging.info(f"serve port: {self.port} max_workers: {self.max_workers}")
        token = "chat-bot-connector"
        authenticator = AuthenticationInterceptor(
            'authorization', token,
            grpc.StatusCode.UNAUTHENTICATED, 'Access denied!'
        )
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.max_workers),
            interceptors=(authenticator,),
        )
        add_ConnectorServicer_to_server(self.connector, server)
        server.add_insecure_port(f"[::]:{self.port}")
        server.start()
        logging.info(f"Server started port: {self.port}")
        server.wait_for_termination()
