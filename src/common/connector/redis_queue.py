import pickle
import logging

from src.common.factory import EngineClass
from src.common.interface import IConnector
from src.common.queue.redis import RedisQueue


class RedisQueueConnector(EngineClass, IConnector):
    TAG = "redis_queue_connector"

    def __init__(self, send_key="SEND", **kwargs) -> None:
        self.conn = RedisQueue(**kwargs)
        self.send_key = send_key

    def close(self):
        self.conn.close()

    def send(self, data, at: str):
        if at not in ["be", "fe"]:
            raise Exception(f"send at {at} must use 'be' or 'fe'")
        data = pickle.dumps(data)
        send_key = f"{at.upper()}_{self.send_key}"
        return self.conn.put(send_key, data)

    def recv(self, at: str):
        if at not in ["be", "fe"]:
            raise Exception(f"recv at {at} must use 'be' or 'fe'")
        send_key = f"FE_{self.send_key}"
        if at == "fe":
            send_key = f"BE_{self.send_key}"
        res = self.conn.get(send_key)
        if res is None:
            return None

        return pickle.loads(res)
