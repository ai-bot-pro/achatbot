import os
import logging

import unittest
from apipeline.pipeline.pipeline import Pipeline, FrameDirection
from apipeline.pipeline.task import PipelineTask, PipelineParams
from apipeline.pipeline.runner import PipelineRunner
from apipeline.frames.data_frames import DataFrame, TextFrame, Frame
from apipeline.frames.control_frames import EndFrame
from apipeline.processors.frame_processor import FrameProcessor

from src.processors.speech.asr.asr_processor import AsrProcessor
from src.common.types import DailyParams, DailyTranscriptionSettings
from src.common.logger import Logger
from src.transports.daily import DailyTransport
from src.types.frames.data_frames import TranscriptionFrame
from src.common.session import Session
from src.common.types import SessionCtx
from src.cmd.init import Env

from dotenv import load_dotenv
load_dotenv(override=True)

"""
ASR_TAG="whisper_asr" \
    python -m unittest test.integration.processors.test_asr_processor.TestASRProcessor
ASR_TAG="whisper_timestamped_asr" \
    python -m unittest test.integration.processors.test_asr_processor.TestASRProcessor
ASR_TAG="whisper_faster_asr" \
    python -m unittest test.integration.processors.test_asr_processor.TestASRProcessor
ASR_TAG="whisper_transformers_asr" \
    ASR_MODEL_NAME_OR_PATH=./models/openai/whisper-base \
    python -m unittest test.integration.processors.test_asr_processor.TestASRProcessor
ASR_TAG="whisper_mlx_asr" \
    python -m unittest test.integration.processors.test_asr_processor.TestASRProcessor
ASR_TAG="sense_voice_asr" \
    ASR_LANG=zn \
    ASR_MODEL_NAME_OR_PATH=./models/FunAudioLLM/SenseVoiceSmall \
    python -m unittest test.integration.processors.test_asr_processor.TestASRProcessor
ASR_TAG="whisper_groq_asr" \
    ASR_LANG=zh \
    ASR_MODEL_NAME_OR_PATH=whisper-large-v3 \
    python -m unittest test.integration.processors.test_asr_processor.TestASRProcessor
"""


class TranscriptionLogger(FrameProcessor):

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            print(f"Transcription: {frame.text}")


class TestASRProcessor(unittest.IsolatedAsyncioTestCase):

    @classmethod
    def setUpClass(cls):
        Logger.init(logging.INFO, is_file=False)
        cls.room_url = os.getenv("DAILY_ROOM_URL", "https://weedge.daily.co/chat-bot")
        cls.room_token = os.getenv("DAILY_ROOM_TOKEN", None)

    @classmethod
    def tearDownClass(cls):
        pass

    async def asyncSetUp(self):
        transport = DailyTransport(self.room_url,
                                   self.room_token,
                                   "Transcription bot",
                                   DailyParams(audio_in_enabled=True))

        asr = Env.initASREngine()
        session = Session(**SessionCtx("test_client_id", 16000, 2).__dict__)
        asr_processor = AsrProcessor(asr=asr, session=session)

        tl_porcessor = TranscriptionLogger()

        pipeline = Pipeline([
            transport.input_processor(),
            asr_processor,
            tl_porcessor])

        self.task = PipelineTask(
            pipeline,
            PipelineParams(
                llow_interruptions=True,
                enable_metrics=True,
            )
        )

    async def asyncTearDown(self):
        pass

    async def test_asr(self):
        runner = PipelineRunner()
        await runner.run(self.task)
