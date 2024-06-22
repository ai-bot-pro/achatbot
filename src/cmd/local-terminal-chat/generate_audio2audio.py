r"""
TQDM_DISABLE=True python -m src.cmd.local-terminal-chat.generate_audio2audio > ./log/std_out.log

TQDM_DISABLE=True RECORDER_TAG=wakeword_rms_recorder python -m src.cmd.local-terminal-chat.generate_audio2audio > ./log/std_out.log

KMP_DUPLICATE_LIB_OK=TRUE TQDM_DISABLE=True RECORDER_TAG=wakeword_rms_recorder ASR_TAG=whisper_faster_asr ASR_MODEL_NAME_OR_PATH=./models/Systran/faster-whisper-base python -m src.cmd.local-terminal-chat.generate_audio2audio > ./log/std_out.log
"""
import multiprocessing
import multiprocessing.connection
import logging

from src.common.logger import Logger
from src.common.connector.multiprocessing_pipe import MultiprocessingPipeConnector
from src.cmd.be import run_be
from src.cmd.fe import run_fe


# global logging
Logger.init(logging.INFO, is_file=True, is_console=False)


def main():
    mp_conn = MultiprocessingPipeConnector()

    # BE
    be_init_event = multiprocessing.Event()
    c = multiprocessing.Process(
        target=run_be, args=(mp_conn, be_init_event))
    c.start()
    be_init_event.wait()

    # FE
    run_fe(mp_conn)

    c.join()
    c.close()

    mp_conn.close()


if __name__ == "__main__":
    main()
