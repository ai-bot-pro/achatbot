import multiprocessing
import multiprocessing.connection


from src.common.interface import IConnector


class MultiprocessingPipeConnector(IConnector):
    def __init__(self) -> None:
        self.fe_conn, self.be_conn = multiprocessing.Pipe()

    def close(self):
        self.fe_conn.close()
        self.be_conn.close()

    def send(self, data, _to: str):
        if _to not in ["be", "fe"]:
            raise Exception(f"send to {_to} must use 'be' or 'fe'")
        if _to == "fe":
            return self.fe_conn.send(data)
        if _to == "be":
            return self.be_conn.send(data)

    def recv(self, _from: str):
        if _from not in ["be", "fe"]:
            raise Exception(f"recv from {_from} must use 'be' or 'fe'")
        if _from == "fe":
            return self.fe_conn.recv()
        if _from == "be":
            return self.be_conn.recv()
