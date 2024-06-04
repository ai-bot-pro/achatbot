import logging


class Logger():
    @staticmethod
    def init(level=logging.INFO):
        # Create a logger
        logger = logging.getLogger()
        logger.setLevel(level)  # Set the root logger's level

        # Create a file handler and set its level
        file_handler = logging.FileHandler('chat-bot.log')
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
