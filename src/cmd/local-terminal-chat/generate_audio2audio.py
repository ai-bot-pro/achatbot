r"""
TQDM_DISABLE=True python -m src.cmd.local-terminal-chat.generate_audio2audio > std_out.log
TQDM_DISABLE=True RECORDER_TAG=wakeword_rms_recorder python -m src.cmd.local-terminal-chat.generate_audio2audio > std_out.log
"""
import multiprocessing
import multiprocessing.connection
import logging

from src.common.logger import Logger
from src.cmd.be import run_be
from src.cmd.fe import run_fe

# global logging for fork processes
logger = Logger.init(logging.INFO, is_file=True, is_console=False)


def main():
    parent_conn, child_conn = multiprocessing.Pipe()

    # BE
    be_init_event = multiprocessing.Event()
    c = multiprocessing.Process(
        target=run_be, args=(child_conn, be_init_event))
    c.start()
    be_init_event.wait()

    # FE
    run_fe(parent_conn)

    c.join()
    c.close()

    parent_conn.close()
    child_conn.close()


if __name__ == "__main__":
    main()
