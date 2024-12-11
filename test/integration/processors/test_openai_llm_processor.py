import asyncio
import os
import logging
from typing import Any, Awaitable, Callable
import uuid

import unittest

from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineTask, PipelineParams
from apipeline.pipeline.runner import PipelineRunner
from apipeline.processors.logger import FrameLogger
from apipeline.frames.data_frames import TextFrame
from apipeline.frames.control_frames import EndFrame
from apipeline.frames.sys_frames import MetricsFrame

from src.common.utils.time import time_now_iso8601
from src.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAIAssistantContextAggregator,
    OpenAIUserContextAggregator,
)
from src.processors.llm.base import LLMProcessor
from src.processors.llm.openai_llm_processor import OpenAIGroqLLMProcessor, OpenAILLMProcessor
from src.processors.vision.vision_processor import MockVisionProcessor, VisionProcessor
from src.common.session import Session
from src.common.types import SessionCtx
from src.core.llm import LLMEnvInit
from src.common.logger import Logger
from src.types.frames.data_frames import LLMMessagesFrame, TranscriptionFrame
from src.types.frames.control_frames import UserStartedSpeakingFrame, UserStoppedSpeakingFrame

from dotenv import load_dotenv

load_dotenv()


"""
python -m unittest test.integration.processors.test_openai_llm_processor.TestProcessor.test_run


MODEL=llama-3.1-70b-versatile \
    python -m unittest test.integration.processors.test_openai_llm_processor.TestProcessor.test_run

BASE_URL=https://api.together.xyz/v1 MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo \
    python -m unittest test.integration.processors.test_openai_llm_processor.TestProcessor.test_run
"""


class TestProcessor(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)
        cls.max_function_call_cn = int(os.environ.get("MAX_FUNCTION_CALL_CN", "3"))
        cls.messages = [
            {
                "role": "system",
                # "content": "You are a weather assistant. Use the get_weather function to retrieve weather information for a given location."
                "content": "You are a helpful assistant who converses with a user and answers questions. Respond concisely to general questions.  Your response will be turned into speech so use only simple words and punctuation.\n  You have access to two tools: get_weather and describe_image.  You can respond to questions about the weather using the get_weather tool.\n  You can answer questions about the user's video stream using the describe_image tool.\n Some examples of phrases that indicate you should use the describe_image tool are: \n - What do you see?  \n - What's in the video? \n - Can you describe the video?\n - Tell me about what you see.\n  - Tell me something interesting about what you see.\n  - What's happening in the video?\n  If you need to use a tool, simply use the tool. Do not tell the user the tool you are using. Be brief and concise.\n Please communicate in Chinese",
            }
        ]
        cls.tools = [
            {
                "function": {
                    "description": "Get the current weather in a given location",
                    "name": "get_weather",
                    "parameters": {
                        "properties": {
                            "location": {
                                "description": "The city and state, e.g. San Francisco, CA",
                                "type": "string",
                            }
                        },
                        "required": ["location"],
                        "type": "object",
                    },
                },
                "type": "function",
            },
        ]

    @classmethod
    def tearDownClass(cls):
        pass

    async def get_weather(
        self,
        function_name: str,
        tool_call_id: str,
        arguments: Any,
        llm: LLMProcessor,
        context: OpenAILLMContext,
        result_callback: Callable[[Any], Awaitable[None]],
    ):
        location = arguments["location"]
        logging.info(
            f"function_name:{function_name}, tool_call_id:{tool_call_id},"
            f"arguments:{arguments}, llm:{llm}, context:{context}"
        )
        # just a mock response
        # add result to assistant context
        self.get_weather_call_cn += 1
        if self.max_function_call_cn > self.get_weather_call_cn:
            await result_callback(f"The weather in {location} is currently 72 degrees and sunny.")
        else:
            self.get_weather_call_cn = 0

    def get_vision_llm_processor(self) -> VisionProcessor:
        if self.vision_tag == "mock_vision_processor":
            return MockVisionProcessor(self.mock_text)

        session = Session(**SessionCtx(uuid.uuid4()).__dict__)
        llm = LLMEnvInit.initLLMEngine()
        return VisionProcessor(llm=llm, session=session)

    def get_openai_llm_processor(self) -> LLMProcessor:
        # default use openai llm processor
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("BASE_URL", "https://api.groq.com/openai/v1")
        model = os.environ.get("MODEL", "llama3-groq-70b-8192-tool-use-preview")
        if "groq" in base_url:
            # https://console.groq.com/docs/models
            api_key = os.environ.get("GROQ_API_KEY")
            llm_processor = OpenAIGroqLLMProcessor(
                model=model,
                base_url=base_url,
                api_key=api_key,
            )
            return llm_processor
        elif "together" in base_url:
            # https://docs.together.ai/docs/chat-models
            api_key = os.environ.get("TOGETHER_API_KEY")
        llm_processor = OpenAILLMProcessor(
            model=model,
            base_url=base_url,
            api_key=api_key,
        )
        return llm_processor

    async def asyncSetUp(self):
        self.llm_context = OpenAILLMContext()
        self.llm_context.set_messages(self.messages)
        self.llm_context.set_tools(self.tools)
        llm_user_ctx_aggr = OpenAIUserContextAggregator(self.llm_context)
        llm_assistant_ctx_aggr = OpenAIAssistantContextAggregator(llm_user_ctx_aggr)

        llm_processor = self.get_openai_llm_processor()
        llm_processor.register_function("get_weather", self.get_weather)
        self.get_weather_call_cn = 0

        pipeline = Pipeline(
            [
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

    async def test_run(self):
        runner = PipelineRunner()
        await self.task.queue_frames(
            [
                UserStartedSpeakingFrame(),
                TranscriptionFrame(
                    "What's the weather like in New York today? ", "", time_now_iso8601(), "en"
                ),
                UserStoppedSpeakingFrame(),
                EndFrame(),
            ]
        )
        await runner.run(self.task)
        msgs = self.llm_context.get_messages()
        print(msgs)
        print("get_weather_call_cn", self.get_weather_call_cn)
        self.assertLess(self.get_weather_call_cn, self.max_function_call_cn)
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
