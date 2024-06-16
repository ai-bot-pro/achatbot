import os
import logging
import asyncio

import unittest

from src.common.logger import Logger
from src.common.types import LOG_DIR
from src.common.queue.redis import RedisQueue

r"""
REDIS_PASSWORD=*** python -m unittest test.common.queue.test_redis
"""


class TestRedisQueue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.REDIS_HOST = os.getenv(
            "REDIS_HOST", "redis-12259.c240.us-east-1-3.ec2.redns.redis-cloud.com")
        cls.REDIS_PORT = os.getenv("REDIS_PORT", "12259")
        Logger.init(logging.DEBUG)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.queue = RedisQueue(host=self.REDIS_HOST, port=self.REDIS_PORT)

    def tearDown(self):
        self.queue.client.flushdb()
        self.queue.client.close()

    def test_get(self):
        asyncio.run(self.queue.put("test_get", b"test"))
        res = asyncio.run(self.queue.get("test_get"))
        print(res)
        self.assertGreater(len(res), 0)

    def test_put(self):
        res = asyncio.run(self.queue.put("test_set", b"test"))
        print(res)
        self.assertGreater(res, 0)
