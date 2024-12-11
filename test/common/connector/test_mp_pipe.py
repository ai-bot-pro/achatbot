import logging

import os
import unittest

from src.common.connector.multiprocessing_pipe import MultiprocessingPipeConnector
from src.common.logger import Logger

r"""
python -m unittest test.common.connector.test_mp_pipe
"""


class TestMultiprocessingPipeConnector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Logger.init(os.getenv("LOG_LEVEL", "debug").upper(), is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.connector = MultiprocessingPipeConnector()

    def tearDown(self):
        self.connector.close()

    def test_send_recv(self):
        print("be send")
        self.connector.send((1, 2, 3), "be")
        print("fe recv")
        v1, v2, v3 = self.connector.recv("fe")
        print(v1, v2, v3)
        self.assertEqual(v1, 1)
        self.assertEqual(v2, 2)
        self.assertEqual(v3, 3)
