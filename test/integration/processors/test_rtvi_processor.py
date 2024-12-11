import os
import logging

import unittest
import uuid
from apipeline.pipeline.pipeline import Pipeline, FrameDirection
from apipeline.pipeline.task import PipelineTask, PipelineParams
from apipeline.pipeline.runner import PipelineRunner
from apipeline.frames.data_frames import Frame, TextFrame
from apipeline.frames.sys_frames import StopTaskFrame
from apipeline.processors.frame_processor import FrameProcessor

from src.processors.speech.tts.tts_processor import TTSProcessor
from src.common.session import Session
from src.common.types import SessionCtx
from src.modules.speech.tts import TTSEnvInit
from src.processors.llm.openai_llm_processor import OpenAILLMProcessor
from src.processors.rtvi.rtvi_processor import RTVIProcessor
from src.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAIAssistantContextAggregator,
    OpenAIUserContextAggregator,
)
from src.common.logger import Logger
from src.types.frames.data_frames import TranscriptionFrame
from src.types.frames.control_frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from src.common.utils.time import time_now_iso8601
from apipeline.processors.logger import FrameLogger

from dotenv import load_dotenv

load_dotenv()

"""
python -m unittest test.integration.processors.test_rtvi_processor.TestRTVIProcessor
"""


class MockProcessor(FrameProcessor):
    def __init__(self, name):
        super().__init__(name=name)
        self.token: list[str] = []
        # Start collecting tokens when we see the start frame
        self.start_collecting = False
        self.answer: list[list] = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self.start_collecting = True
        elif isinstance(frame, UserStartedSpeakingFrame):
            self.token = []
        elif isinstance(frame, TextFrame) and self.start_collecting:
            self.token.append(frame.text)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self.answer.append(self.token)
        elif isinstance(frame, LLMFullResponseEndFrame):
            self.start_collecting = False

        await self.push_frame(frame, direction)


class TestRTVIProcessor(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    async def asyncSetUp(self):
        self.session = Session(**SessionCtx(uuid.uuid4()).__dict__)

        llm_context = OpenAILLMContext()
        llm_user_ctx_aggr = OpenAIUserContextAggregator(llm_context)
        llm_assistant_ctx_aggr = OpenAIAssistantContextAggregator(llm_user_ctx_aggr)
        llm_processor = OpenAILLMProcessor(
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.together.xyz/v1"),
            model=os.environ.get("OPENAI_MODEL", "Qwen/Qwen2-72B-Instruct"),
            api_key=os.environ.get("TOGETHER_API_KEY"),
            # api_key=os.environ.get("GROQ_API_KEY"),
        )

        self.mock_processor = MockProcessor("stream_tokens")

        tts = TTSEnvInit.initTTSEngine()
        tts_processor = TTSProcessor(tts=tts, session=self.session)

        pipeline = Pipeline(
            [
                FrameLogger(),
                RTVIProcessor(),
                FrameLogger(),
                # accumulate TranscriptionFrame,InterimTranscriptionFrame
                # UserStartedSpeakingFrame, UserStoppedSpeakingFrame,
                llm_user_ctx_aggr,
                FrameLogger(),
                llm_processor,  # OpenAILLMContextFrame, LLMMessagesFrame
                FrameLogger(),
                self.mock_processor,  # TextFrame
                FrameLogger(),
                # TextFrame match_endofsentence to say
                tts_processor,
                FrameLogger(),
                # accumulate TextFrame
                # LLMFullResponseStartFrame, LLMFullResponseEndFrame
                # FunctionCallInProgressFrame, FunctionCallResultFrame,
                # to check run llm again with llm_user_ctx_aggr
                llm_assistant_ctx_aggr,
                FrameLogger(),
            ]
        )

        self.task = PipelineTask(pipeline, PipelineParams())

    async def asyncTearDown(self):
        pass

    async def test_run(self):
        runner = PipelineRunner()
        await self.task.queue_frames(
            [
                UserStartedSpeakingFrame(),
                TranscriptionFrame("你好", "", time_now_iso8601(), "zh"),
                UserStoppedSpeakingFrame(),
                # StopTaskFrame(),
            ]
        )
        await runner.run(self.task)

        logging.info(f"answer: {self.mock_processor.answer}")

        self.assertGreater(len(self.mock_processor.token), 0)
        self.assertEqual(len(self.mock_processor.answer), 1)
