import os
import logging

import unittest
from apipeline.pipeline.pipeline import Pipeline, FrameDirection
from apipeline.pipeline.task import PipelineTask, PipelineParams
from apipeline.pipeline.runner import PipelineRunner
from apipeline.frames.data_frames import Frame
from apipeline.processors.frame_processor import FrameProcessor

from src.processors.ffmpeg_pipe_processor import AudioRawFrame, FFMPEGPipeProcessor, FFmpegPiper
from src.modules.speech.vad_analyzer.silero import SileroVADAnalyzer
from src.common.types import DailyParams
from src.common.logger import Logger
from src.transports.daily import DailyTransport

from dotenv import load_dotenv

load_dotenv(override=True)

"""
python -m unittest test.integration.processors.test_ffmpeg_pipe_processor.TestProcessor
"""


class TranscriptionLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            print(f"get AudioRawFrame Frame: {str(frame)}")


class TestProcessor(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)
        cls.room_url = os.getenv("DAILY_ROOM_URL", "https://weedge.daily.co/chat-room")
        cls.room_token = os.getenv("DAILY_ROOM_TOKEN", None)
        cls.vad_enabled = bool(os.getenv("DAILY_VAD_ENABLED", ""))

    @classmethod
    def tearDownClass(cls):
        pass

    async def asyncSetUp(self):
        daily_pramams: DailyParams = DailyParams(
            audio_in_enabled=True,
            transcription_enabled=False,
        )
        if self.vad_enabled is True:
            daily_pramams.vad_enabled = True
            daily_pramams.vad_analyzer = SileroVADAnalyzer()
            daily_pramams.vad_audio_passthrough = True

        self.transport = DailyTransport(
            self.room_url,
            self.room_token,
            "Piper Bot",
            daily_pramams,
        )

        self.ffmpeg_piper = FFmpegPiper(
            in_sample_rate=daily_pramams.audio_in_sample_rate,
            in_channels=daily_pramams.audio_in_channels,
            in_sample_width=daily_pramams.audio_in_sample_width,
            out_sample_rate=daily_pramams.audio_out_sample_rate,
            out_channels=daily_pramams.audio_out_channels,
            out_sample_width=daily_pramams.audio_out_sample_width,
        )

    async def asyncTearDown(self):
        pass

    async def test_asr(self):
        ffmpeg_pipe_processor = FFMPEGPipeProcessor(ffmpeg_piper=self.ffmpeg_piper, min_chunk_s=0.5)
        tl_processor = TranscriptionLogger()

        pipeline = Pipeline([self.transport.input_processor(), ffmpeg_pipe_processor, tl_processor])

        self.task = PipelineTask(
            pipeline,
            PipelineParams(
                allow_interruptions=False,
            ),
        )
        runner = PipelineRunner()
        await runner.run(self.task)
