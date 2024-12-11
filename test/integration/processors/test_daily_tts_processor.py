import asyncio
import os
import logging

import unittest
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.runner import PipelineRunner
from apipeline.pipeline.task import PipelineTask, PipelineParams
from apipeline.frames.control_frames import EndFrame
from apipeline.frames.data_frames import TextFrame

from src.common.types import DailyParams
from src.services.help.daily_rest import DailyRESTHelper
from src.common.logger import Logger
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.modules.speech.tts import TTSEnvInit
from src.transports.daily import DailyTransport

from dotenv import load_dotenv

load_dotenv(override=True)

r"""
DAILY_ROOM_URL=https://weedge.daily.co/chat-room \
    python -m unittest test.integration.processors.test_daily_tts_processor.TestTTSProcessor
"""


DAILY_API_URL = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")
DAILY_API_KEY = os.getenv("DAILY_API_KEY", "")
ROOM_EXPIRE_TIME = 30 * 60  # 30 minutes
ROOM_TOKEN_EXPIRE_TIME = 30 * 60  # 30 minutes


class TestTTSProcessor(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)
        cls.room_url = os.getenv("DAILY_ROOM_URL", "https://weedge.daily.co/chat-room")

    @classmethod
    def tearDownClass(cls):
        pass

    async def asyncSetUp(self):
        self.daily_rest_helper = DailyRESTHelper(DAILY_API_KEY, DAILY_API_URL)
        pass

    async def asyncTearDown(self):
        pass

    async def test_tts_daily_output(self):
        tts = TTSEnvInit.initTTSEngine()
        stream_info = tts.get_stream_info()

        bot_name = "my test bot"
        transport = DailyTransport(
            self.room_url,
            None,
            bot_name,
            DailyParams(
                audio_out_enabled=True,
                audio_out_sample_rate=stream_info["rate"],
            ),
        )
        tts_processor = TTSProcessor(tts=tts)
        # await tts_processor.set_voice("zh-CN-YunjianNeural")
        # await tts_processor.set_voice("zh-CN-YunxiNeural")
        # await tts_processor.set_voice("zh-CN-YunxiaNeural")
        # await tts_processor.set_voice("zh-CN-YunyangNeural")
        # await tts_processor.set_voice("zh-CN-liaoning-XiaobeiNeural")
        # await tts_processor.set_voice("zh-CN-shaanxi-XiaoniNeural")
        # await tts_processor.set_voice("zh-TW-HsiaoYuNeural")
        # await tts_processor.set_voice("zh-HK-WanLungNeural")
        # await tts_processor.set_voice("zh-HK-HiuMaanNeural")
        # await tts_processor.set_voice("zh-CN-XiaoyiNeural")
        await tts_processor.set_voice("zh-CN-XiaoxiaoNeural")

        task = PipelineTask(
            Pipeline([tts_processor, transport.output_processor()]),
            params=PipelineParams(allow_interruptions=True),
        )

        # Register an event handler so we can play the audio
        # when the participant joins.
        @transport.event_handler("on_participant_joined")
        async def on_new_participant_joined(transport, participant):
            participant_name = bot_name
            if "userName" in participant["info"]:
                participant_name = participant["info"]["userName"]
            await task.queue_frames(
                [
                    TextFrame(f"你好，Hello there. {participant_name}"),
                    TextFrame("你是一个中国人。"),
                    # TextFrame(f"一名中文助理，请用中文简短回答，回答限制在5句话内。"),
                    # TextFrame(f"我是Andrej Karpathy，我在YouTube上发布关于机器学习和深度学习的视频。如果你有任何关于这些视频的疑问或需要帮助，请告诉我！"),
                    EndFrame(),
                ]
            )

        runner = PipelineRunner()
        await runner.run(task)
