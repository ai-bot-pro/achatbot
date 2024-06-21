import asyncio
import pickle

from src.common.interface import IConnector
from src.common.queue.redis import RedisQueue


class RedisQueueConnector(IConnector):
    def __init__(self,
                 fe_send_key="FE_SEND",
                 be_send_key="BE_SEND", **kwargs) -> None:
        self.fe_conn = RedisQueue(**kwargs)
        self.be_conn = RedisQueue(**kwargs)
        self.fe_send_key = fe_send_key
        self.be_send_key = be_send_key

    def close(self):
        self.fe_conn.close()
        self.be_conn.close()

    def send(self, data, _to: str):
        if _to not in ["be", "fe"]:
            raise Exception(f"send to {_to} must use 'be' or 'fe'")
        data = pickle.dumps(data)
        if _to == "fe":
            return asyncio.run(self.fe_conn.put(self.fe_send_key, data))
        if _to == "be":
            return asyncio.run(self.be_conn.put(self.be_send_key, data))

    def recv(self, _from: str):
        if _from not in ["be", "fe"]:
            raise Exception(f"recv from {_from} must use 'be' or 'fe'")
        if _from == "fe":
            res = self.fe_conn.get(self.be_send_key)
        if _from == "be":
            res = self.be_conn.get(self.fe_send_key)

        return pickle.loads(res)
