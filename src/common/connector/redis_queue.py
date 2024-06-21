import asyncio
import pickle

from src.common.interface import IConnector
from src.common.queue.redis import RedisQueue


class RedisQueueConnector(IConnector):
    def __init__(self,
                 fe_send_key="FE_SEND",
                 be_send_key="BE_SEND", **kwargs) -> None:
        self.conn = RedisQueue(**kwargs)
        self.fe_send_key = fe_send_key
        self.be_send_key = be_send_key

    def close(self):
        self.conn.close()

    def send(self, data, at: str):
        if at not in ["be", "fe"]:
            raise Exception(f"send at {at} must use 'be' or 'fe'")
        data = pickle.dumps(data)
        if at == "fe":
            return asyncio.run(self.conn.put(self.fe_send_key, data))
        if at == "be":
            return asyncio.run(self.conn.put(self.be_send_key, data))

    def recv(self, at: str):
        if at not in ["be", "fe"]:
            raise Exception(f"recv at {at} must use 'be' or 'fe'")
        if at == "fe":
            res = asyncio.run(self.conn.get(self.be_send_key))
        if at == "be":
            res = asyncio.run(self.conn.get(self.fe_send_key))
        if res is None:
            return None

        return pickle.loads(res)
