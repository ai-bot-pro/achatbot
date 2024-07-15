import traceback
import logging
import threading
import queue
import os

import grpc

from src.common.grpc.idl.connector_pb2 import (
    ConnectStreamRequest, ConnectStreamResponse
)
from src.common.grpc.idl.connector_pb2_grpc import ConnectorStub
from src.common.grpc.interceptors.authentication_client import add_authentication
from src.common.logger import Logger


class StreamClient():
    def __init__(self, in_q: queue.Queue, out_q: queue.Queue,
                 token="chat-bot-connenctor", target="localhost:50052") -> None:
        self.in_q = in_q
        self.out_q = out_q
        self.token = token
        self.target = target
        self.is_active = False
        self.connect_thread = None

    def start(self):
        if self.connect_thread is None or self.connect_thread.is_alive() is False:
            self.is_active = True
            self.connect_thread = threading.Thread(target=self._connect, args=())
            self.connect_thread.start()

    def stop(self):
        self.is_active = False
        if self.connect_thread is not None and self.connect_thread.is_alive():
            self.connect_thread.join()

    def _connect(self):
        authentication = add_authentication('authorization', self.token)
        channel = grpc.insecure_channel(self.target)
        channel = grpc.intercept_channel(channel, authentication)
        stub = ConnectorStub(channel)

        def iterator():
            while True:
                frame = self.in_q.get()
                if frame is None:
                    break
                yield ConnectStreamRequest(frame=frame)
        try:
            response = stub.ConnectStream(iterator)
            self.out_q.put(response.frame)
        except Exception as e:
            logging.error(e)
            logging.error(traceback.format_exc())

    def close(self):
        self.in_q.put(None)
        self.stop()
