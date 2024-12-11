import unittest

from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models import FakeStreamingListLLM
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.runner import PipelineRunner
from apipeline.pipeline.task import PipelineTask, PipelineParams
from apipeline.processors.frame_processor import FrameProcessor
from apipeline.frames.sys_frames import StopTaskFrame

from src.types.frames.control_frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from src.types.frames.data_frames import TextFrame, TranscriptionFrame
from src.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator,
)
from src.processors.ai_frameworks.langchain_processor import LangchainProcessor


r"""
python -m unittest test.integration.processors.test_langchain_processor.TestLangchainProcessor
"""


class MockProcessor(FrameProcessor):
    def __init__(self, name):
        super().__init__(name=name)
        self.token: list[str] = []
        # Start collecting tokens when we see the start frame
        self.start_collecting = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self.start_collecting = True
        elif isinstance(frame, TextFrame) and self.start_collecting:
            self.token.append(frame.text)
        elif isinstance(frame, LLMFullResponseEndFrame):
            self.start_collecting = False

        await self.push_frame(frame, direction)


class TestLangchainProcessor(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.expected_response = "Hello dear human"
        self.fake_llm = FakeStreamingListLLM(responses=[self.expected_response])

    async def asyncTearDown(self):
        pass

    async def test_langchain(self):
        messages = [("system", "Say hello to {name}"), ("human", "{input}")]
        prompt = ChatPromptTemplate.from_messages(messages).partial(name="Thomas")
        chain = prompt | self.fake_llm
        proc = LangchainProcessor(chain=chain)

        tma_in = LLMUserResponseAggregator(messages)
        tma_out = LLMAssistantResponseAggregator(messages)
        mock_proc = MockProcessor("token_collector")

        pipeline = Pipeline(
            [
                tma_in,
                proc,
                mock_proc,
                tma_out,
            ]
        )

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=False))
        await task.queue_frames(
            [
                UserStartedSpeakingFrame(),
                TranscriptionFrame(text="Hi World", user_id="user", timestamp="now"),
                UserStoppedSpeakingFrame(),
                StopTaskFrame(),
            ]
        )

        runner = PipelineRunner()
        await runner.run(task)
        self.assertEqual("".join(mock_proc.token), self.expected_response)
