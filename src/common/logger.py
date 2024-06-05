import os
import logging

from .types import LOG_DIR


class Logger():
    @staticmethod
    def init(level=logging.INFO, app_name="chat-bot", log_dir=LOG_DIR):
        os.makedirs(log_dir, exist_ok=True)
        # Create a logger
        logger = logging.getLogger()
        logger.setLevel(level)  # Set the root logger's level
        logger.name = app_name

        # Create a file handler and set its level
        log_path = os.path.join(log_dir, f"{app_name}.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)

        # Create a console handler and set its level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # set log formatter
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        file_handler.setFormatter(logging.Formatter(log_format))
        console_handler.setFormatter(logging.Formatter(log_format))

        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger
