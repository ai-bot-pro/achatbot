import os
import logging
from dataclasses import dataclass, field

from src.common.factory import EngineClass


@dataclass
class RedisInfoArgs:
    host: str | None = field(
        default=None,
        metadata={"help": "redis host, Default: None"},
    )
    port: str | None = field(
        default=None,
        metadata={"help": "redis port, Default: None"},
    )
    password: str | None = field(
        default=None,
        metadata={"help": "redis password, Default: None"},
    )
    db: int = field(
        default=0,
        metadata={"help": "redis db, Default: 0"},
    )
    prefix_key: str = field(
        default="B:FRAMES",
        metadata={"help": "redis prefix key, Default: 'B:FRAMES'"},
    )


class RedisQueue(EngineClass):
    TAG = "queue_redis"

    def __init__(self, **kwargs) -> None:
        import redis

        self._args = RedisInfoArgs(**kwargs)
        redis_host = self._args.host if self._args.host else os.getenv("REDIS_HOST", "localhost")
        redis_port = self._args.port if self._args.port else os.getenv("REDIS_PORT", "6379")
        redis_password = os.getenv("REDIS_PASSWORD", "")
        self.client = redis.Redis(
            host=redis_host, port=redis_port, password=redis_password, db=self._args.db
        )
        if self.client.ping():
            logging.debug(f"connect redis:{redis_host}:{redis_port} success")
        else:
            logging.debug(f"connect redis:{redis_host}:{redis_port} fail!")

    def _get_key(self, key: str) -> str:
        return f"{self._args.prefix_key}:{key}"

    def get(self, key: str, timeout_s=0) -> bytes:
        key = self._get_key(key)
        logging.debug(f"get key: {key}")
        res = self.client.blpop(key, timeout=timeout_s)
        if res is None:
            return None
        return res[1]

    def put(self, key: str, data: bytes):
        key = self._get_key(key)
        logging.debug(f"put key: {key}")
        return self.client.rpush(key, data)

    def close(self):
        self.client.close()
