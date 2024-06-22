import logging
import asyncio
import threading
import sys

import uuid

from src.common import interface
from src.common.session import Session
from src.common.utils.audio_utils import save_audio_to_file
from src.common.types import SessionCtx, RECORDS_DIR
from src.cmd import init


def run_fe(conn: interface.IConnector):
    start_record_event = threading.Event()
    play_t = threading.Thread(target=loop_play,
                              args=(conn, start_record_event))
    play_t.start()
    record_t = threading.Thread(target=loop_record,
                                args=(conn, start_record_event))
    record_t.start()

    record_t.join()
    play_t.join()


def loop_record(conn: interface.IConnector, e: threading.Event):
    recorder = init.initRecorderEngine()
    sid = uuid.uuid4()
    session = Session(**SessionCtx(sid).__dict__)
    session.ctx.waker = init.initWakerEngine()
    logging.info(f"loop_record starting with session ctx: {session.ctx}")
    print("start loop_record...", flush=True, file=sys.stderr)
    while True:
        try:
            print(f"-- chat round {session.chat_round} --",
                  flush=True, file=sys.stdout)
            print("\nme >> ", end="", flush=True, file=sys.stderr)
            e.clear()

            frames = recorder.record_audio(session)
            if len(frames) == 0:
                continue
            data = b''.join(frames)
            conn.send(("RECORD_FRAMES", data, session), 'fe')
            asyncio.run(save_audio_to_file(
                data, session.get_file_name(), audio_dir=RECORDS_DIR))
            session.increment_file_counter()
            session.increment_chat_round()

            e.wait()
        except Exception as ex:
            logging.warning(
                f"loop_record Exception {ex} sid:{session.ctx.client_id}")


def loop_play(conn: interface.IConnector, e: threading.Event):
    player = init.initPlayerEngine()
    print("start loop_play...", flush=True, file=sys.stderr)
    llm_gen_segments = 0
    while True:
        try:
            res = conn.recv('fe')
            if res is None:
                continue
            msg, recv_data, session = res
            if msg is None or msg.lower() == "stop":
                break
            if msg == "PLAY_FRAMES":
                session.ctx.state["tts_chunk"] = recv_data
                player.play_audio(session)
                e.set()
                llm_gen_segments = 0
            elif msg == "LLM_GENERATE_TEXT":
                if llm_gen_segments == 0:
                    bot_name = session.ctx.state["bot_name"] if "bot_name" in session.ctx.state else "bot"
                    logging.info(f"bot_name: {bot_name}")
                    print(f"\n{bot_name} >> ", end="",
                          flush=True, file=sys.stderr)
                print(recv_data.strip(), end="", flush=True, file=sys.stderr)
                llm_gen_segments += 1
            elif msg == "LLM_GENERATE_DONE":
                print("\n", end="", flush=True, file=sys.stderr)
                llm_gen_segments = 0
            elif msg == "ASR_TEXT":
                print(recv_data.strip(), end="", flush=True, file=sys.stderr)
            elif msg == "ASR_TEXT_DONE":
                print("\n", end="", flush=True, file=sys.stderr)
            elif msg == "BE_EXCEPTION":
                print(f"\nBE exception: {recv_data.strip()}",
                      end="", flush=True, file=sys.stderr)
                e.set()
                llm_gen_segments = 0
            else:
                logging.warning(f"unsupport msg {msg}")
        except Exception as ex:
            logging.warning(f"loop_play Exception {ex}")
            e.set()
            llm_gen_segments = 0
