import logging
import os


class RedisQueue:
    Q_PREFIX_KEY = "B:FRAMES"

    def __init__(self, host=None, port=None, password=None, db=0) -> None:
        import redis
        redis_host = host if host else os.getenv(
            "REDIS_HOST", "localhost")
        redis_port = port if port else os.getenv("REDIS_PORT", "6379")
        redis_password = password if password else os.getenv(
            "REDIS_PASSWORD", "")
        self.client = redis.Redis(
            host=redis_host, port=redis_port, password=redis_password, db=db)
        if self.client.ping():
            logging.debug(f"connect redis:{redis_host}:{redis_port} success")
        else:
            logging.debug(f"connect redis:{redis_host}:{redis_port} fail!")

    def _get_key(self, key: str) -> str:
        return f"{self.Q_PREFIX_KEY}:{key}"

    async def get(self, key: str) -> bytes:
        key = self._get_key(key)
        q_len = self.client.llen(key)
        if q_len > 10:
            self.client.expire(key, 1)
        frames = self.client.blpop(key, timeout=0.1)
        return frames

    async def put(self, key: str, data: bytes):
        key = self._get_key(key)
        return self.client.rpush(key, data)

    def close(self):
        self.client.close()
        pass
