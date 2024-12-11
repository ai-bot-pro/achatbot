import os
import logging
import asyncio

import unittest

from src.common.logger import Logger
from src.common.session import Session
from src.common.types import LOG_DIR, SessionCtx
from src.common.connector.redis_queue import RedisQueueConnector


from dotenv import load_dotenv

load_dotenv(override=True)


r"""
REDIS_PASSWORD=*** python -m unittest test.common.connector.test_redis_queue
"""


class TestRedisQueue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.REDIS_HOST = os.getenv(
            "REDIS_HOST", "redis-12259.c240.us-east-1-3.ec2.redns.redis-cloud.com"
        )
        cls.REDIS_PORT = os.getenv("REDIS_PORT", "12259")
        Logger.init(os.getenv("LOG_LEVEL", "debug").upper(), is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.connector = RedisQueueConnector(
            send_key="TEST_SEND", host=self.REDIS_HOST, port=self.REDIS_PORT
        )

    def tearDown(self):
        self.connector.close()

    def test_send_recv(self):
        self.connector.send((1, 2, 3), "be")
        v1, v2, v3 = self.connector.recv("fe")
        print(v1, v2, v3)
        self.assertEqual(v1, 1)
        self.assertEqual(v2, 2)
        self.assertEqual(v3, 3)

    def test_send_recv_session(self):
        s = Session(**SessionCtx("test_sid").__dict__)
        s.chat_history.append("hello")
        self.connector.send((1, 2, s), "be")
        v1, v2, session = self.connector.recv("fe")
        print(v1, v2, session)
        print(session.chat_history)
        self.assertEqual(v1, 1)
        self.assertEqual(v2, 2)
        self.assertIsNotNone(session)

        session.chat_history.append("world")
        self.connector.send((1, 2, session), "fe")
        v1, v2, s = self.connector.recv("be")
        print(v1, v2, s)
        print(s.chat_history)
        self.assertEqual(v1, 1)
        self.assertEqual(v2, 2)
        self.assertIsNotNone(s)
