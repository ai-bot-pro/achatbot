import logging
import time

from .base import BaseBot, register_rtvi_bots


@register_rtvi_bots.register
class DummyBot(BaseBot):
    def run(self):
        try:
            logging.info("dummy bot run")
            time.sleep(60)
            logging.info("dummy bot over")
        except KeyboardInterrupt:
            logging.info("Ctrl-C detected. Exiting!")
