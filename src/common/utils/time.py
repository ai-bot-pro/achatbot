import time


def get_current_formatted_time():
    timestamp = time.time()
    formatted_time = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
    return formatted_time
