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
