import os
import logging

from .types import LOG_DIR


def getDefaultFormatter():
    # set log formatter
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(funcName)s - %(message)s"
    formatter = logging.Formatter(log_format)
    return formatter


def addLoggingLevel(levelName, levelNum, methodName=None):
    # Adopted from https://stackoverflow.com/a/35804945/1691778
    # Adds a new logging method to the logging module

    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
        raise AttributeError("{} already defined in logging module".format(levelName))
    if hasattr(logging, methodName):
        raise AttributeError("{} already defined in logging module".format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
        raise AttributeError("{} already defined in logger class".format(methodName))

    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)


# Create the TRACE level
addLoggingLevel("TRACE", logging.DEBUG - 5)


class Logger:
    """
    set root or app logger at once
    """

    inited = False
    logger = None

    @staticmethod
    def init(
        level: int | str = logging.INFO,
        app_name="chat-bot",
        log_dir=LOG_DIR,
        is_file=True,
        is_console=True,
        is_root_logger=True,
        is_re_init=False,
    ):
        if Logger.inited and is_re_init is False and Logger.logger is not None:
            return Logger.logger
        os.makedirs(log_dir, exist_ok=True)
        if is_root_logger:
            # get root logger to set, global logger, use logging
            logger = logging.getLogger()
            logger.setLevel(level)  # Set the root logger's level
            logger.name = app_name
        else:
            # create app logger, u can use init logger
            logger = logging.getLogger(name=app_name)
            logger.setLevel(level)  # Set the app logger's level

        if is_file:
            # Create a file handler and set its level
            log_path = os.path.join(log_dir, f"{app_name}.log")
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(level)
            file_handler.setFormatter(getDefaultFormatter())
            logger.addHandler(file_handler)

        if is_console:
            # Create a console handler and set its level
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(getDefaultFormatter())
            logger.addHandler(console_handler)

        Logger.inited = True
        Logger.logger = logger

        return logger
