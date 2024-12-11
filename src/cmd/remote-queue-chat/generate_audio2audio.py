import os
import logging
import argparse

import uuid

from src.cmd.fe import TerminalChatClient
from src.cmd.be import Audio2AudioChatWorker as ChatWorker
from src.common.connector.redis_queue import RedisQueueConnector
from src.common.logger import Logger


r"""

REDIS_PASSWORD=$redis_pwd RUN_OP=fe python -m src.cmd.remote-queue-chat.generate_audio2audio > ./log/fe_std_out.log
REDIS_PASSWORD=$redis_pwd RUN_OP=be TQDM_DISABLE=True python -m src.cmd.remote-queue-chat.generate_audio2audio > ./log/be_std_out.log

# with wakeword
REDIS_PASSWORD=$redis_pwd RUN_OP=fe RECORDER_TAG=wakeword_rms_recorder python -m src.cmd.remote-queue-chat.generate_audio2audio > ./log/fe_std_out.log
REDIS_PASSWORD=$redis_pwd RUN_OP=be TQDM_DISABLE=True python -m src.cmd.remote-queue-chat.generate_audio2audio > ./log/be_std_out.log

TTS_TAG=tts_cosy_voice \
  REDIS_PASSWORD=$redis_pwd \
  RUN_OP=fe \
  RECORDER_TAG=wakeword_rms_recorder \
  python -m src.cmd.remote-queue-chat.generate_audio2audio > ./log/fe_std_out.log

# sense_voice (asr) -> qwen (llm) -> cosy_voice (tts)
RUN_OP=be \
  TQDM_DISABLE=True \
  REDIS_PASSWORD=$redis_pwd \
  ASR_TAG=sense_voice_asr \
  ASR_LANG=zn \
  N_GPU_LAYERS=33 FLASH_ATTN=1 \
  LLM_MODEL_NAME=qwen \
  LLM_MODEL_PATH=./models/qwen1_5-7b-chat-q8_0.gguf \
  ASR_MODEL_NAME_OR_PATH=./models/FunAudioLLM/SenseVoiceSmall \
  TTS_TAG=tts_cosy_voice \
  python -m src.cmd.remote-queue-chat.generate_audio2audio > ./log/be_std_out.log
"""


# global logging
Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=True, is_console=False)


def main():
    conn = RedisQueueConnector(
        send_key=os.getenv("SEND_KEY", "SEND"),
        host=os.getenv("REDIS_HOST", "redis-12259.c240.us-east-1-3.ec2.redns.redis-cloud.com"),
        port=os.getenv("REDIS_PORT", "12259"),
    )

    op = os.getenv("RUN_OP", "fe")
    if op == "fe":
        client = TerminalChatClient()
        client.run(conn)
    else:
        ChatWorker().run(conn)
    conn.close()


if __name__ == "__main__":
    main()
