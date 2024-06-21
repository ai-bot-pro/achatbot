import os
import logging
import argparse

from src.cmd.fe import run_fe
from src.cmd.be import run_be
from src.common.connector.redis_queue import RedisQueueConnector
from src.common.logger import Logger


r"""

REDIS_PASSWORD=*** RUN_OP=fe python -m src.cmd.remote-queue-chat.generate_audio2audio > ./log/fe_std_out.log 
REDIS_PASSWORD=*** RUN_OP=be TQDM_DISABLE=True python -m src.cmd.remote-queue-chat.generate_audio2audio > ./log/be_std_out.log 

# with wakeword
REDIS_PASSWORD=*** RUN_OP=fe RECORDER_TAG=wakeword_rms_recorder python -m src.cmd.remote-queue-chat.generate_audio2audio > ./log/fe_std_out.log 
REDIS_PASSWORD=*** RUN_OP=be TQDM_DISABLE=True python -m src.cmd.remote-queue-chat.generate_audio2audio > ./log/be_std_out.log
"""


# global logging
Logger.init(logging.INFO, is_file=True, is_console=False)


def main():
    conn = RedisQueueConnector(
        host=os.getenv(
            "REDIS_HOST",
            "redis-12259.c240.us-east-1-3.ec2.redns.redis-cloud.com"),
        port=os.getenv("REDIS_PORT", "12259"))

    op = os.getenv("RUN_OP", "fe")
    if op == "be":
        run_be(conn)
    else:
        run_fe(conn)
    conn.close()


if __name__ == "__main__":
    main()
