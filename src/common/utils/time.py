import time
import datetime


def get_current_formatted_time():
    timestamp = time.time()
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
    return formatted_time


def time_now_iso8601() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")
