import asyncio
from io import BytesIO
import os
import logging

import unittest
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineTask, PipelineParams
from apipeline.pipeline.runner import PipelineRunner
from apipeline.frames import EndFrame, AudioRawFrame
from apipeline.processors.output_processor import OutputFrameProcessor
import librosa
import torch
import numpy as np
import soundfile as sf

from src.common.utils.wav import read_wav_to_bytes
from src.common.logger import Logger
from src.common.types import TEST_DIR, MODELS_DIR
from src.transports.daily import DailyTransport
from src.common.types import DailyParams
from src.types.frames.control_frames import (
    AvatarArgsUpdateFrame,
    AvatarLanguageUpdateFrame,
    AvatarModelUpdateFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from src.types.avatar.musetalk import AvatarMuseTalkConfig


from dotenv import load_dotenv


load_dotenv(override=True)

"""
# download model weights
huggingface-cli download weege007/musetalk --local-dir ./models/weege007/musetalk

python -m unittest test.integration.processors.test_musetalk_avatar_processor.TestMusetalkProcessor.test_gen

WEIGHT_DIR=./models/weege007/musetalk \
    python -m unittest test.integration.processors.test_musetalk_avatar_processor.TestMusetalkProcessor.test_gen

DAILY_ROOM_URL=https://weedge.daily.co/jk5g4mFlZkPHvOyaEZe5 \
    WEIGHT_DIR=./models/weege007/musetalk \
    python -m unittest test.integration.processors.test_musetalk_avatar_processor.TestMusetalkProcessor.test_gen
DAILY_ROOM_URL=https://weedge.daily.co/jk5g4mFlZkPHvOyaEZe5 DEBUG=true \
    WEIGHT_DIR=./models/weege007/musetalk \
    python -m unittest test.integration.processors.test_musetalk_avatar_processor.TestMusetalkProcessor.test_gen
DAILY_ROOM_URL=https://weedge.daily.co/jk5g4mFlZkPHvOyaEZe5 DEBUG=true \
    WEIGHT_DIR=./models/weege007/musetalk \
    MATERIAL_VIDEO_PATH=./deps/MuseTalk/data/video/yongen.mp4 \
    FORCE_PREPARATION=true \
    python -m unittest test.integration.processors.test_musetalk_avatar_processor.TestMusetalkProcessor.test_gen

DAILY_ROOM_URL=https://weedge.daily.co/jk5g4mFlZkPHvOyaEZe5 DEBUG=true \
    WEIGHT_DIR=./models/weege007/musetalk \
    MATERIAL_VIDEO_PATH=./deps/MuseTalk/data/video/yongen.mp4 \
    FORCE_PREPARATION=true \
    FPS=25 BATCH_SIZE=20 GEN_BATCH_SIZE=5 \    
    python -m unittest test.integration.processors.test_musetalk_avatar_processor.TestMusetalkProcessor.test_gen
"""


class TestMusetalkProcessor(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # use for gpu to test
        assert torch.cuda.is_available()

        cls._debug = os.getenv("DEBUG", "false").lower() == "true"
        cls.force_preparation = os.getenv("FORCE_PREPARATION", "false").lower() == "true"

        # https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/asr/test_audio/asr_example_zh.wav
        audio_file = os.path.join(TEST_DIR, "audio_files/asr_example_zh.wav")
        cls.audio_file = os.getenv("AUDIO_FILE", audio_file)
        cls.data_bytes, cls.sr = read_wav_to_bytes(cls.audio_file)

        cls.room_url = os.getenv("DAILY_ROOM_URL", "https://weedge.daily.co/chat-room")

        cls.version = os.getenv("VERSION", "v15")  # v1, v15
        cls.result_dir = os.getenv("RESULT_DIR", "./results")
        cls.model_dir = os.getenv("WEIGHT_DIR", os.path.join(MODELS_DIR, "weege007/musetalk"))
        cls.gpu_id = int(os.getenv("GPU_ID", "0"))
        cls.batch_size = int(os.getenv("BATCH_SIZE", "20"))
        cls.gen_batch_size = int(os.getenv("GEN_BATCH_SIZE", "5"))
        cls.fps = int(os.getenv("FPS", "25"))
        cls.material_video_path = os.getenv(
            "MATERIAL_VIDEO_PATH", "./deps/MuseTalk/data/video/sun.mp4"
        )

        cls.sleep_to_end_time_s = int(os.getenv("SLEEP_TO_END_TIME_S", "30"))

        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)

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
        from src.processors.avatar.musetalk_avatar_processor import MusetalkAvatarProcessor
        from src.modules.avatar.musetalk import MusetalkAvatar

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

        avatar = MusetalkAvatar(
            avatar_id="avator_test",
            material_video_path=self.material_video_path,
            force_preparation=self.force_preparation,
            version=self.version,
            result_dir=self.result_dir,
            model_dir=self.model_dir,
            gpu_id=self.gpu_id,
            debug=self._debug,
            batch_size=self.batch_size,
            gen_batch_size=self.gen_batch_size,
            fps=self.fps,
        )
        avatar.load()
        config = AvatarMuseTalkConfig(
            debug=self._debug,
            debug_save_handler_audio=self._debug,
            algo_audio_sample_rate=self.sr,
            output_audio_sample_rate=self.sr,
            input_audio_slice_duration=1,
            fps=avatar.fps,
            batch_size=avatar.gen_batch_size,
        )
        musetalkProcessor = MusetalkAvatarProcessor(avatar=avatar, config=config)
        pipeline = Pipeline(
            [
                musetalkProcessor,
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
