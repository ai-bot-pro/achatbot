from functools import wraps
import time
import logging


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logging.debug(f"Function {func.__name__} took {execution_time:.3f}s to execute.")
        return result

    return wrapper


def to_timestamp(t: int, comma: bool = False, msec: int = 10):
    """
    whisper cpp time to timestamp
    """
    msec = int(t * msec)
    hours = int(msec / (1000 * 60 * 60))
    msec = int(msec - hours * (1000 * 60 * 60))
    minutes = int(msec / (1000 * 60))
    msec = int(msec - minutes * (1000 * 60))
    sec = int(msec / 1000)
    msec = int(msec - sec * 1000)

    return "{:02d}:{:02d}:{:02d}{}{:03d}".format(hours, minutes, sec, "," if comma else ".", msec)
