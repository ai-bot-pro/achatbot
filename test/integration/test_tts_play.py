import os
import logging
import threading
import traceback

import unittest
import uuid

from src.common.connector.redis_queue import RedisQueueConnector
from src.common.logger import Logger
from src.common.session import Session
from src.common.types import SessionCtx
from src.common import interface
from src.common.factory import EngineClass

if os.getenv("INIT_TYPE", "env") == "yaml_config":
    from src.cmd.init import YamlConfig as init
else:
    from src.cmd.init import Env as init

r"""
REDIS_PASSWORD=*** TTS_TAG=tts_g python -m unittest test.integration.test_tts_play.TestTTSPlay.test_tts

REDIS_PASSWORD=*** TTS_TAG=tts_g python -m unittest test.integration.test_tts_play.TestTTSPlay.test_filter_special_chars
"""


class TestTTSPlay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)

    @classmethod
    def tearDownClass(cls):
        pass

    def on_play_end(self, session: Session):
        logging.info(f"play end with session.ctx {session.ctx}")

    def on_play_chunk(self, session: Session, sub_chunk: bytes):
        logging.info(
            f"play chunk with session.ctx.state {session.ctx.state} sub_chunk_len {len(sub_chunk)}"
        )

    def setUp(self):
        self.sid = str(uuid.uuid4())
        self.session = Session(**SessionCtx(self.sid).__dict__)
        self.tts: interface.ITts | EngineClass = init.initTTSEngine()
        self.conn = RedisQueueConnector(
            send_key=os.getenv("SEND_KEY", "SEND"),
            host=os.getenv("REDIS_HOST", "redis-12259.c240.us-east-1-3.ec2.redns.redis-cloud.com"),
            port=os.getenv("REDIS_PORT", "12259"),
        )

        self.player = init.initPlayerEngine()
        self.player.set_args(on_play_end=self.on_play_end)
        self.player.set_args(on_play_chunk=self.on_play_chunk)
        self.player.start(self.session)

    def tearDown(self):
        self.player.stop(self.session)
        self.conn.close()

    def test_filter_special_chars(self):
        test_case = [
            {"text": "这是个测试字符串", "expect": "这是个测试字符串"},
            {"text": "这是个测试字符串？", "expect": "这是个测试字符串？"},
            {"text": "这是个测试字符串？！？！」}", "expect": "这是个测试字符串？"},
            {"text": "？", "expect": ""},
            {"text": "【结束】", "expect": "【结束】"},
            {"text": "\n\n【结束】", "expect": "【结束】"},
            # {"text": "请问有什么问题或主题需要我提供信息或者帮助吗？"},
            # {"text": "请告诉我您的需求，我会尽力为您提供最佳答案。"},
            # {"text": "祝您好运！"},
        ]
        for item in test_case:
            res = self.tts.filter_special_chars(item["text"])
            print(f"res:{res}")
            self.assertEqual(res, item["expect"])

    def test_tts(self):
        play_t = threading.Thread(target=self.loop_recv)
        play_t.start()

        send_t = threading.Thread(target=self.send)
        send_t.start()

        send_t.join()
        play_t.join()

    def send(self):
        test_case = [
            {"text": "你好！"},
            {"text": "很高兴能帮助您。"},
            {"text": "。!>}]"},
            {"text": "请问有什么问题或主题需要我提供信息或者帮助吗？"},
            {"text": "请告诉我您的需求，我会尽力为您提供最佳答案。"},
            {"text": "祝您好运！"},
        ]
        for item in test_case:
            self.session.ctx.state["tts_text"] = item["text"]
            audio_iter = self.tts.synthesize_sync(self.session)
            for i, chunk in enumerate(audio_iter):
                logging.info(f"synthesize audio {i} chunk {len(chunk)}")
                if len(chunk) > 0:
                    self.conn.send(("PLAY_FRAMES", chunk, self.session), "be")
        self.conn.send(("PLAY_FRAMES_DONE", "", self.session), "be")

    def loop_recv(self):
        logging.info(f"start loop_recv with {self.player.TAG} ...")
        while True:
            try:
                res = self.conn.recv("fe")
                if res is None:
                    continue

                msg, recv_data, self.session = res
                if msg is None or msg.lower() == "stop":
                    break
                logging.info(f"msg: {msg} recv_data_len: {len(recv_data)}")

                if msg == "PLAY_FRAMES":
                    self.session.ctx.state["tts_chunk"] = recv_data
                    self.player.play_audio(self.session)
                elif msg == "PLAY_FRAMES_DONE":
                    self.player.stop(self.session)
                    self.player.start(self.session)
                    break
                else:
                    logging.warning(f"unsupport msg {msg}")
            except Exception as ex:
                ex_trace = traceback.format_exc()
                logging.warning(f"loop_recv Exception {ex}, trace: {ex_trace}")
