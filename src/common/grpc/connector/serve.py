from concurrent import futures
import threading
import logging
import queue

import grpc

from src.common.grpc.idl.connector_pb2_grpc import (
    ConnectorServicer,
    add_ConnectorServicer_to_server,
)
from src.common.grpc.idl.connector_pb2 import ConnectStreamResponse
from src.common.grpc.interceptors.authentication_server import AuthenticationInterceptor


class Connector(ConnectorServicer):
    def __init__(self, in_q: queue.Queue, out_q: queue.Queue) -> None:
        super().__init__()
        self.in_q = in_q
        self.out_q = out_q
        self.is_active = False
        self.recv_thread = None

    def ConnectStream(self, request_iterator, context):
        def recv_to_queue(request_iterator):
            logging.info("ConnectStream api recv_to_queue thread start")
            try:
                for request in request_iterator:
                    if request.frame is None:
                        logging.debug("connectStream request frame is None, break")
                        self.is_active = False
                        break
                    logging.debug(f"be in_q put ----> len(request.frame):{len(request.frame)}")
                    self.in_q.put(request.frame)
            except Exception as e:
                logging.error(f"ConnectStream api recv_to_queue error:{e}")

        # if self.recv_thread is None or self.recv_thread.is_alive() is False:
        self.recv_thread = threading.Thread(target=recv_to_queue, args=(request_iterator,))
        self.recv_thread.start()
        self.is_active = True

        logging.info("ConnectStream api wait out_q frame to yield")
        while self.is_active:
            frame = self.out_q.get()
            if frame is None:
                logging.debug("connectStream response frame is None, break")
                break
            logging.debug(f"be out_q put ----> len(frame):{len(frame)}")
            yield ConnectStreamResponse(frame=frame)

    def close(self):
        self.out_q.put(None)
        self.is_active = False
        if self.recv_thread is not None and self.recv_thread.is_alive():
            self.recv_thread.join()


class StreamServe:
    def __init__(
        self,
        in_q: queue.Queue,
        out_q: queue.Queue,
        token="chat-bot-connenctor",
        port="50052",
        max_workers=10,
    ) -> None:
        self.token = token
        self.port = port
        self.max_workers = max_workers
        self.connector = Connector(in_q, out_q)
        self.serve_thread: threading.Thread = None

        logging.info(f"serve port: {self.port} max_workers: {self.max_workers}")
        authenticator = AuthenticationInterceptor(
            "authorization", self.token, grpc.StatusCode.UNAUTHENTICATED, "Access denied!"
        )
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.max_workers),
            interceptors=(authenticator,),
        )
        add_ConnectorServicer_to_server(self.connector, server)
        server.add_insecure_port(f"[::]:{self.port}")
        self.server = server

    def start(self):
        if self.serve_thread is None or self.serve_thread.is_alive() is False:
            self.serve_thread = threading.Thread(target=self._serve, args=())
            self.serve_thread.start()

    def stop(self):
        logging.debug("Server stoping...")
        self.server.stop(1).wait()
        logging.info("Server stop")
        if self.serve_thread is not None or self.serve_thread.is_alive():
            self.serve_thread.join()
            logging.info("serve_thread stop")

    def close(self):
        self.connector.close()
        self.stop()

    def _serve(self) -> None:
        self.server.start()
        logging.info(f"Server started port: {self.port}")
        self.server.wait_for_termination()
