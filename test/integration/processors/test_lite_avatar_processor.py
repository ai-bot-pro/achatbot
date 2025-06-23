import asyncio
import os
import logging

import unittest
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineTask, PipelineParams
from apipeline.pipeline.runner import PipelineRunner
from apipeline.frames import EndFrame, AudioRawFrame
from apipeline.processors.output_processor import OutputFrameProcessor
import torch

from src.common.utils.wav import read_wav_to_bytes
from src.processors.avatar.lite_avatar_processor import LiteAvatarProcessor
from src.common.logger import Logger
from src.common.types import TEST_DIR
from src.modules.avatar.lite_avatar import LiteAvatar
from src.types.avatar.lite_avatar import MODELS_DIR, AvatarInitOption
from src.transports.daily import DailyTransport
from src.common.types import DailyParams
from src.types.frames.control_frames import (
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)

from dotenv import load_dotenv


load_dotenv(override=True)

"""
# download model weights
huggingface-cli download weege007/liteavatar --local-dir ./models/weege007/liteavatar

python -m unittest test.integration.processors.test_lite_avatar_processor.TestLiteAvatarProcessor.test_gen

WEIGHT_DIR=./models/weege007/liteavatar \
    python -m unittest test.integration.processors.test_lite_avatar_processor.TestLiteAvatarProcessor.test_gen

WEIGHT_DIR=./models/weege007/liteavatar \
    AVATAR_NAME=20250612/P1-64AzfrJY037WpS69RiUMw \
    python -m unittest test.integration.processors.test_lite_avatar_processor.TestLiteAvatarProcessor.test_gen

DAILY_ROOM_URL=https://weedge.daily.co/jk5g4mFlZkPHvOyaEZe5 \
    python -m unittest test.integration.processors.test_lite_avatar_processor.TestLiteAvatarProcessor.test_gen

DAILY_ROOM_URL=https://weedge.daily.co/jk5g4mFlZkPHvOyaEZe5 \
    WEIGHT_DIR=./models/weege007/liteavatar \
    AVATAR_NAME=20250612/P1-64AzfrJY037WpS69RiUMw \
    python -m unittest test.integration.processors.test_lite_avatar_processor.TestLiteAvatarProcessor.test_gen
"""


class TestLiteAvatarProcessor(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)

        # https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/asr/test_audio/asr_example_zh.wav
        audio_file = os.path.join(TEST_DIR, "audio_files/asr_example_zh.wav")
        cls.audio_file = os.getenv("AUDIO_FILE", audio_file)
        cls.data_bytes, cls.sr = read_wav_to_bytes(cls.audio_file)

        weight_dir = os.path.join(MODELS_DIR, "weege007/liteavatar")
        cls.weight_dir = os.getenv("WEIGHT_DIR", weight_dir)

        cls.avatar_name = os.getenv("AVATAR_NAME", "20250408/sample_data")

        cls.room_url = os.getenv("DAILY_ROOM_URL", "https://weedge.daily.co/chat-room")

        cls.use_gpu = True if torch.cuda.is_available() else False

        cls.sleep_to_end_time_s = int(os.getenv("SLEEP_TO_END_TIME_S", "20"))

    @classmethod
    def tearDownClass(cls):
        pass

    async def out_cb(self, frame):
        # await asyncio.sleep(1)
        logging.info(f"sink_callback print frame: {frame}")

    async def end_task(self):
        await asyncio.sleep(self.sleep_to_end_time_s)
        await self.task.queue_frame(EndFrame())

    async def asyncSetUp(self):
        bot_name = "avatar-bot"
        transport = DailyTransport(
            self.room_url,
            None,
            bot_name,
            DailyParams(
                audio_out_enabled=True,
                audio_out_sample_rate=self.sr,
                camera_out_enabled=True,
                camera_out_width=1024,
                camera_out_height=1408,
                camera_out_is_live=True,
            ),
        )

        liteAvatarProcessor = LiteAvatarProcessor(
            LiteAvatar(
                **AvatarInitOption(
                    audio_sample_rate=self.sr,
                    video_frame_rate=25,
                    avatar_name=self.avatar_name,
                    is_show_video_debug_text=True,
                    enable_fast_mode=False,
                    use_gpu=self.use_gpu,
                    weight_dir=self.weight_dir,
                    is_flip=False,
                ).__dict__
            ),
        )
        pipeline = Pipeline(
            [
                liteAvatarProcessor,
                # OutputFrameProcessor(cb=self.out_cb),
                transport.output_processor(),
            ]
        )

        self.task = PipelineTask(
            pipeline,
            PipelineParams(
                allow_interruptions=False,
                enable_metrics=False,
            ),
        )
        self.runner = PipelineRunner()

        self.end_task: asyncio.Task = asyncio.get_event_loop().create_task(self.end_task())

    async def asyncTearDown(self):
        self.end_task.cancel()
        await self.end_task

    async def test_gen(self):
        # ctrl + C to stop
        await self.task.queue_frames(
            [
                UserStartedSpeakingFrame(),
                AudioRawFrame(
                    audio=self.data_bytes,
                    sample_rate=self.sr,
                ),
                UserStoppedSpeakingFrame(),
                # EndFrame(),
            ]
        )
        await self.runner.run(self.task)
