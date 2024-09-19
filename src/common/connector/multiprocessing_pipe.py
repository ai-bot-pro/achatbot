import multiprocessing


from src.common.interface import IConnector


class MultiprocessingPipeConnector(IConnector):
    def __init__(self) -> None:
        self.fe_conn, self.be_conn = multiprocessing.Pipe()

    def close(self):
        self.fe_conn.close()
        self.be_conn.close()

    def send(self, data, at: str):
        if at not in ["be", "fe"]:
            raise Exception(f"send at {at} must use 'be' or 'fe'")
        if at == "fe":
            return self.fe_conn.send(data)
        if at == "be":
            return self.be_conn.send(data)

    def recv(self, at: str, timeout: float | None = None):
        if at not in ["be", "fe"]:
            raise Exception(f"recv at {at} must use 'be' or 'fe'")
        if at == "fe":
            if timeout is None:
                return self.fe_conn.recv()
            if timeout and self.fe_conn.poll(timeout):
                return self.fe_conn.recv()
        if at == "be":
            if timeout is None:
                return self.be_conn.recv()
            if timeout and self.be_conn.poll(timeout):
                return self.be_conn.recv()
        return None
