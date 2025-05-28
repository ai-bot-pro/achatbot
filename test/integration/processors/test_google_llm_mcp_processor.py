import os
import shutil
import uuid
import logging
from typing import Any, Awaitable, Callable

import unittest
from PIL import Image

from apipeline.pipeline.pipeline import Pipeline, FrameDirection
from apipeline.pipeline.task import PipelineTask, PipelineParams
from apipeline.pipeline.runner import PipelineRunner
from apipeline.processors.frame_processor import FrameProcessor
from apipeline.processors.logger import FrameLogger
from apipeline.frames.data_frames import TextFrame
from apipeline.frames.control_frames import EndFrame
from apipeline.frames.sys_frames import Frame, MetricsFrame
from mcp import StdioServerParameters

from src.processors.aggregators.vision_image_frame import VisionImageFrameAggregator
from src.processors.llm.google_llm_processor import GoogleAILLMProcessor
from src.common.utils.time import time_now_iso8601
from src.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAIAssistantContextAggregator,
    OpenAIUserContextAggregator,
)
from src.processors.llm.base import LLMProcessor
from src.processors.vision.vision_processor import MockVisionProcessor, VisionProcessor
from src.common.session import Session
from src.common.types import TEST_DIR, SessionCtx
from src.core.llm import LLMEnvInit
from src.common.logger import Logger
from src.types.frames.data_frames import TranscriptionFrame, UserImageRawFrame, VisionImageRawFrame
from src.types.frames.control_frames import (
    UserImageRequestFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from src.services.mcp_client import MCPClient

from dotenv import load_dotenv

load_dotenv()


"""
python -m unittest test.integration.processors.test_google_llm_mcp_processor.TestProcessor.test_run_text

python -m unittest test.integration.processors.test_google_llm_mcp_processor.TestProcessor.test_run_mcp

python -m unittest test.integration.processors.test_google_llm_mcp_processor.TestProcessor.test_run_all
"""


class MockUserImageRawFrameProcessor(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # await super().process_frame(frame,direction)
        if isinstance(frame, UserImageRequestFrame):
            img_file = os.path.join(TEST_DIR, "img_files", "03-Confusing-Pictures.jpg")
            img_file = os.getenv("IMG_FILE", img_file)
            img = Image.open(img_file)
            frame = UserImageRawFrame(
                user_id=frame.user_id,
                image=img.tobytes(),
                size=img.size,
                format=img.format,
                mode=img.mode,
            )
        await self.push_frame(frame)


class TestProcessor(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)
        cls.max_function_call_cn = int(os.environ.get("MAX_FUNCTION_CALL_CN", "3"))
        cls.uid = str(uuid.uuid4())
        cls.messages = [
            {
                "role": "system",
                "content": "You are a web content summary assistant. Use the fetch_website function to fetch website content for a given url. and summary the website content.",
                # "content": "You are a helpful assistant who converses with a user and answers questions. Respond concisely to general questions.  Your response will be turned into speech so use only simple words and punctuation.\n If you need to use a tool, simply use the tool. Do not tell the user the tool you are using. Be brief and concise.\n Please communicate in Chinese",
            }
        ]

    @classmethod
    def tearDownClass(cls):
        pass

    def get_vision_llm_processor(self) -> VisionProcessor:
        if self.vision_tag == "mock_vision_processor":
            return MockVisionProcessor(self.mock_text)

        session = Session(**SessionCtx(self.uid).__dict__)
        llm = LLMEnvInit.initLLMEngine()
        return VisionProcessor(llm=llm, session=session)

    def get_google_llm_processor(self) -> LLMProcessor:
        api_key = os.environ.get("GOOGLE_API_KEY")
        # https://ai.google.dev/gemini-api/docs/models?hl=zh-cn#model-variations
        # model = os.environ.get("MODEL", "gemini-2.5-flash-preview-04-17")
        model = os.environ.get("MODEL", "gemini-2.0-flash-lite")
        config = {
            "api_key": api_key,
            "model": model,
            "generation_config": {
                "max_output_tokens": 1024,
                "temperature": 0.1,
                "top_p": 0.1,
                "top_k": 40,
                "response_mime_type": "text/plain",
            },
        }
        llm_processor = GoogleAILLMProcessor(**config)
        return llm_processor

    async def asyncSetUp(self):
        try:
            server_params = StdioServerParameters(
                command=shutil.which("python"),
                args=["-m", "mcp_server.cmd.simple.lowlevel"],
                # args=["-m", "mcp_server.cmd.weather.fastmcp"],
                env={
                    # "OPENWEATHERMAP_API_KEY": os.getenv("OPENWEATHERMAP_API_KEY", ""),
                    # "FUNC_WEATHER_TAG": os.getenv("FUNC_WEATHER_TAG", "openweathermap_api"),
                },
            )
            # print(f"{server_params=}")
            mcp_client = MCPClient(
                server_params=server_params,
            )
        except Exception as e:
            logging.error(f"error setting up mcp")
            logging.exception(f"error trace: {e}")
            raise e

        self.llm_context = OpenAILLMContext()
        self.llm_context.set_messages(self.messages)
        llm_user_ctx_aggr = OpenAIUserContextAggregator(self.llm_context)
        llm_assistant_ctx_aggr = OpenAIAssistantContextAggregator(llm_user_ctx_aggr)

        llm_processor = self.get_google_llm_processor()
        print(f"{mcp_client=}")
        tools = await mcp_client.register_tools(llm_processor)
        self.llm_context.set_tools(tools)

        pipeline = Pipeline(
            [
                MockUserImageRawFrameProcessor(),
                VisionImageFrameAggregator(pass_text=True),
                llm_user_ctx_aggr,
                llm_processor,
                llm_assistant_ctx_aggr,
                FrameLogger(include_frame_types=[MetricsFrame]),
            ]
        )
        self.task = PipelineTask(
            pipeline,
            params=PipelineParams(enable_metrics=True),
        )

    async def asyncTearDown(self):
        pass

    async def test_run_text(self):
        runner = PipelineRunner()
        await self.task.queue_frames(
            [
                UserStartedSpeakingFrame(),
                TranscriptionFrame("what's your name? ", self.uid, time_now_iso8601(), "en"),
                UserStoppedSpeakingFrame(),
                EndFrame(),
            ]
        )
        await runner.run(self.task)
        msgs = self.llm_context.get_messages()
        print(msgs)
        self.assertEqual(len(msgs), 3)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user")
        self.assertEqual(msgs[2]["role"], "assistant")

    async def test_run_mcp(self):
        runner = PipelineRunner()
        await self.task.queue_frames(
            [
                UserStartedSpeakingFrame(),
                TranscriptionFrame(
                    # "What's the weather like in New York today? ",
                    "please fetch url: https://www.baidu.com/",
                    self.uid,
                    time_now_iso8601(),
                    "en",
                ),
                UserStoppedSpeakingFrame(),
                EndFrame(),
            ]
        )
        await runner.run(self.task)
        msgs = self.llm_context.get_messages()
        print(msgs)
        self.assertEqual(len(msgs) % 2, 1)
        if len(msgs) == 3:
            self.assertEqual(msgs[0]["role"], "system")
            self.assertEqual(msgs[1]["role"], "user")
            self.assertEqual(msgs[2]["role"], "assistant")
        if len(msgs) == 5:
            self.assertEqual(msgs[0]["role"], "system")
            self.assertEqual(msgs[1]["role"], "user")
            self.assertEqual(msgs[2]["role"], "assistant")
            self.assertEqual(msgs[3]["role"], "tool")
            self.assertEqual(msgs[4]["role"], "assistant")

    async def test_run_all(self):
        runner = PipelineRunner()
        await self.task.queue_frames(
            [
                UserStartedSpeakingFrame(),
                TranscriptionFrame("what's your name? ", self.uid, time_now_iso8601(), "en"),
                UserStoppedSpeakingFrame(),
                UserStartedSpeakingFrame(),
                TranscriptionFrame(
                    "please fetch url: https://www.baidu.com/", self.uid, time_now_iso8601(), "en"
                ),
                UserStoppedSpeakingFrame(),
                EndFrame(),
            ]
        )
        await runner.run(self.task)
        msgs = self.llm_context.get_messages()
        print(msgs)
        # !NOTE: no save image chat context message
        self.assertEqual(len(msgs), 7)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user")
        self.assertEqual(msgs[2]["role"], "assistant")
        self.assertEqual(msgs[3]["role"], "user")
        self.assertEqual(msgs[4]["role"], "assistant")
        self.assertEqual(msgs[5]["role"], "tool")
        self.assertEqual(msgs[6]["role"], "assistant")
