import os
import logging

from .types import LOG_DIR


def getDefaultFormatter():
    # set log formatter
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(funcName)s - %(message)s'
    formatter = logging.Formatter(log_format)
    return formatter


class Logger():

    @staticmethod
    def init(level=logging.INFO, app_name="chat-bot", log_dir=LOG_DIR, console=True):
        os.makedirs(log_dir, exist_ok=True)
        # Create a logger
        logger = logging.getLogger()
        logger.setLevel(level)  # Set the root logger's level
        logger.name = app_name

        # Create a file handler and set its level
        log_path = os.path.join(log_dir, f"{app_name}.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(getDefaultFormatter())
        logger.addHandler(file_handler)

        if console:
            # Create a console handler and set its level
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(getDefaultFormatter())
            logger.addHandler(console_handler)

        return logger
