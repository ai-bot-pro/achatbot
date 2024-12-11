import queue
import pickle

from src.common.interface import IConnector
from src.common.grpc.connector.serve import StreamServe
from src.common.grpc.connector.client import StreamClient


class GrpcStreamClientConnector(IConnector):
    def __init__(self, target="localhost:50052", token="chat-bot-connenctor") -> None:
        self.in_q = queue.Queue()
        self.out_q = queue.Queue()
        self.cli = StreamClient(in_q=self.in_q, out_q=self.out_q, target=target, token=token)
        self.cli.start()

    def send(self, data, at: str = "fe"):
        data = pickle.dumps(data)
        self.in_q.put(data)

    def recv(self, at: str = "fe"):
        res = self.out_q.get()
        if res is None:
            return None

        return pickle.loads(res)

    def close(self):
        self.cli.close()


class GrpcStreamServeConnector(IConnector):
    def __init__(self, port="50052", token="chat-bot-connenctor") -> None:
        self.in_q = queue.Queue()
        self.out_q = queue.Queue()
        self.serve = StreamServe(in_q=self.in_q, out_q=self.out_q, port=port, token=token)
        self.serve.start()

    def send(self, data, at: str = "be"):
        data = pickle.dumps(data)
        self.out_q.put(data)

    def recv(self, at: str = "be"):
        res = self.in_q.get()
        if res is None:
            return None

        return pickle.loads(res)

    def close(self):
        self.serve.close()
