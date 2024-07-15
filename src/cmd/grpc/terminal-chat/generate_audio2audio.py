import os
import logging
import argparse

import uuid

from src.cmd.fe import TerminalChatClient
from src.cmd.be import Audio2AudioChatWorker as ChatWorker
from src.common.connector.grpc_stream import (
    GrpcStreamClientConnector, GrpcStreamServeConnector
)
from src.common.logger import Logger


r"""
## note: be need to run before fe

RUN_OP=be TQDM_DISABLE=True \
    TTS_TAG=tts_edge \
    python -m src.cmd.grpc.terminal-chat.generate_audio2audio > ./log/be_std_out.log

RUN_OP=fe \
    TTS_TAG=tts_edge \
    python -m src.cmd.grpc.terminal-chat.generate_audio2audio > ./log/fe_std_out.log
"""

# global logging
Logger.init(logging.INFO, is_file=True, is_console=False)


def main():
    op = os.getenv("RUN_OP", "fe")
    conn = None
    if op == "fe":
        conn = GrpcStreamClientConnector()
        client = TerminalChatClient()
        client.run(conn)
    else:
        conn = GrpcStreamServeConnector()
        ChatWorker().run(conn)
    conn.close()


if __name__ == "__main__":
    main()
