import os
import logging

import unittest

from src.common.logger import Logger
from src.common.types import LOG_DIR

r"""
python -m unittest test.common.test_logger
"""


class TestLogger(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Logger.init(logging.TRACE)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_logger_print(self):
        logging.trace("test trace")
        logging.debug("test debug")
        logging.info("test info")
        logging.warning("test warning")
        logging.error("test error")
        logging.critical("test critical")
        logging.exception("test exception")
