import os
import logging
import asyncio

import unittest

from src.common.logger import Logger
from src.common.types import LOG_DIR
from src.common.connector.redis_queue import RedisQueueConnector

r"""
REDIS_PASSWORD=*** python -m unittest test.common.connector.test_redis_queue
"""


class TestRedisQueue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.REDIS_HOST = os.getenv(
            "REDIS_HOST", "redis-12259.c240.us-east-1-3.ec2.redns.redis-cloud.com")
        cls.REDIS_PORT = os.getenv("REDIS_PORT", "12259")
        Logger.init(logging.DEBUG, is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.connector = RedisQueueConnector(
            host=self.REDIS_HOST, port=self.REDIS_PORT)

    def tearDown(self):
        self.connector.close()

    def test_send_recv(self):
        self.connector.send((1, 2, 3), 'be')
        v1, v2, v3 = self.connector.recv('fe')
        print(v1, v2, v3)
        self.assertEqual(v1, 1)
        self.assertEqual(v2, 2)
        self.assertEqual(v3, 3)
