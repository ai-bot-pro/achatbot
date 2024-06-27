import logging
import asyncio
import threading
import sys
import traceback
import os


import uuid

from src.common import interface
from src.common.session import Session
from src.common.utils.audio_utils import save_audio_to_file
from src.common.types import SessionCtx, RECORDS_DIR
if os.getenv("INIT_TYPE", 'env') == 'yaml_config':
    from src.cmd.init import YamlConfig as init
else:
    from src.cmd.init import Env as init


class TerminalChatClient:
    def __init__(self) -> None:
        self.sid = uuid.uuid4()
        self.session = Session(**SessionCtx(self.sid).__dict__)

    def run(self, conn: interface.IConnector):
        self.player = init.initPlayerEngine()
        self.recorder = init.initRecorderEngine()
        self.waker = init.initWakerEngine()

        start_record_event = threading.Event()
        play_t = threading.Thread(target=self.loop_play,
                                  args=(conn, start_record_event))
        play_t.start()
        record_t = threading.Thread(target=self.loop_record,
                                    args=(conn, start_record_event))
        record_t.start()

        record_t.join()
        play_t.join()

    def on_wakeword_detected(self, session, data):
        if "bot_name" in session.ctx.state:
            print(f"{session.ctx.state['bot_name']}~ ",
                  end="", flush=True, file=sys.stderr)

    def loop_record(self, conn: interface.IConnector, e: threading.Event):
        self.waker.set_args(on_wakeword_detected=self.on_wakeword_detected)
        logging.info(
            f"loop_record starting with session ctx: {self.session.ctx}")
        print("start loop_record...", flush=True, file=sys.stderr)
        while True:
            try:
                print(f"-- chat round {self.session.chat_round} --",
                      flush=True, file=sys.stdout)
                print("\nme >> ", end="", flush=True, file=sys.stderr)
                e.clear()

                self.session.ctx.waker = self.waker
                frames = self.recorder.record_audio(self.session)
                if len(frames) == 0:
                    continue
                data = b''.join(frames)
                conn.send(("RECORD_FRAMES", data, self.session), 'fe')
                asyncio.run(save_audio_to_file(
                    data, self.session.get_file_name(), audio_dir=RECORDS_DIR))
                self.session.increment_file_counter()
                self.session.increment_chat_round()

                e.wait()
            except Exception as ex:
                logging.warning(
                    f"loop_record Exception {ex} sid:{self.session.ctx.client_id}")

    def loop_play(self, conn: interface.IConnector, e: threading.Event):
        print("start loop_play...", flush=True, file=sys.stderr)
        llm_gen_segments = 0
        while True:
            try:
                res = conn.recv('fe')
                if res is None:
                    continue
                msg, recv_data, self.session = res
                if msg is None or msg.lower() == "stop":
                    break
                if msg == "PLAY_FRAMES":
                    self.session.ctx.state["tts_chunk"] = recv_data
                    self.player.play_audio(self.session)
                    e.set()
                    llm_gen_segments = 0
                elif msg == "LLM_GENERATE_TEXT":
                    if llm_gen_segments == 0:
                        bot_name = self.session.ctx.state["bot_name"] if "bot_name" in self.session.ctx.state else "bot"
                        logging.info(f"bot_name: {bot_name}")
                        print(f"\n{bot_name} >> ", end="",
                              flush=True, file=sys.stderr)
                    print(recv_data.strip(), end="",
                          flush=True, file=sys.stderr)
                    llm_gen_segments += 1
                elif msg == "LLM_GENERATE_DONE":
                    print("\n", end="", flush=True, file=sys.stderr)
                    llm_gen_segments = 0
                elif msg == "ASR_TEXT":
                    print(recv_data.strip(), end="",
                          flush=True, file=sys.stderr)
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
                ex_trace = ''.join(traceback.format_tb(ex.__traceback__))
                logging.warning(f"loop_play Exception {ex}, trace: {ex_trace}")
                e.set()
                llm_gen_segments = 0

    @staticmethod
    def clear_console():
        os.system('clear' if os.name == 'posix' else 'cls')
