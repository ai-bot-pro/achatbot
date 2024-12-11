import logging

import os
import unittest

from src.common.connector.grpc_stream import GrpcStreamClientConnector, GrpcStreamServeConnector
from src.common.logger import Logger
from src.common.interface import IConnector

r"""
python -m unittest test.common.connector.test_grpc_stream
"""


class TestGrpcStreamConnector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Logger.init(os.getenv("LOG_LEVEL", "debug").upper(), is_file=False, is_console=True)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.be_connector = GrpcStreamServeConnector()
        self.fe_connector = GrpcStreamClientConnector()

    def tearDown(self):
        self.be_connector.close()
        self.fe_connector.close()
        pass

    def test_send_recv(self):
        print("be send")
        self.be_connector.send((1, 2, 3), "be")
        print("fe recv")
        v1, v2, v3 = self.fe_connector.recv("fe")
        print(v1, v2, v3)
        self.assertEqual(v1, 1)
        self.assertEqual(v2, 2)
        self.assertEqual(v3, 3)

        print("fe send")
        self.fe_connector.send((3, 2, 1), "fe")
        print("be recv")
        v3, v2, v1 = self.be_connector.recv("be")
        print(v1, v2, v3)
        self.assertEqual(v1, 1)
        self.assertEqual(v2, 2)
        self.assertEqual(v3, 3)
