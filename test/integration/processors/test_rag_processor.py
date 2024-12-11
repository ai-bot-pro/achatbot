import os
import logging

import unittest

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import TiDBVectorStore

from langchain_community.embeddings import JinaEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from apipeline.processors.frame_processor import FrameProcessor
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.frames.sys_frames import StopTaskFrame
from apipeline.pipeline.runner import PipelineRunner

from src.cmd.bots.rag.daily_langchain_rag_bot import DEFAULT_SYSTEM_PROMPT
from src.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator,
)
from src.processors.ai_frameworks.langchain_rag_processor import LangchainRAGProcessor
from src.cmd.bots.rag.helper import get_tidb_url
from src.common.logger import Logger

from src.types.frames.control_frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from src.types.frames.data_frames import TextFrame, TranscriptionFrame

from dotenv import load_dotenv

load_dotenv(override=True)


r"""
python -m unittest test.integration.processors.test_rag_processor.TestRAGLangchainProcessor.test_rag
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


class TestRAGLangchainProcessor(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)
        self.message_store = {}
        self.base_url = os.environ.get("LLM_OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
        self.model = os.environ.get("LLM_OPENAI_MODEL", "llama-3.1-70b-versatile")
        self.lang = os.environ.get("LLM_LANG", "zh")

    async def asyncTearDown(self):
        pass

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.message_store:
            self.message_store[session_id] = ChatMessageHistory()
        logging.info(f"session id: {session_id}, msg: {self.message_store[session_id]}")
        return self.message_store[session_id]

    async def test_rag(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if "groq" in self.base_url:
            api_key = os.environ.get("GROQ_API_KEY")
        elif "together" in self.base_url:
            api_key = os.environ.get("TOGETHER_API_KEY")
        llm = ChatOpenAI(
            model=self.model,
            base_url=self.base_url,
            api_key=api_key,
            temperature=0.8,
            top_p=0.7,
            max_retries=3,
            max_tokens=1024,
        )

        embed_model_name = "jina-embeddings-v2-base-en"
        if self.lang == "zh":
            embed_model_name = "jina-embeddings-v2-base-zh"
        print(embed_model_name)
        vectorstore = TiDBVectorStore(
            connection_string=get_tidb_url(),
            embedding_function=JinaEmbeddings(
                jina_api_key=os.getenv("JINA_API_KEY"),
                model_name=embed_model_name,
            ),
            table_name="AndrejKarpathy",
            distance_strategy=os.getenv("TIDB_VSS_DISTANCE_STRATEGY", "cosine"),
        )
        score_threshold = 0.8
        if self.lang == "zh":
            score_threshold = 0.5
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": score_threshold},
        )

        system_prompt = DEFAULT_SYSTEM_PROMPT
        if self.lang == "zh":
            system_prompt += " You must reply, Please communicate in Chinese"
        logging.info(f"use system prompt: {system_prompt}")
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_prompt
                    + """ \
                {context}""",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, answer_prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        # chain = prompt | rag
        rag_history_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            history_messages_key="chat_history",
            input_messages_key="input",
            output_messages_key="answer",
        )
        langchain_processor = LangchainRAGProcessor(chain=rag_history_chain)

        tma_in = LLMUserResponseAggregator()
        tma_out = LLMAssistantResponseAggregator()
        mock_proc = MockProcessor("token_collector")

        pipeline = Pipeline(
            [
                tma_in,
                langchain_processor,
                mock_proc,
                tma_out,
            ]
        )

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=False))

        text_frames = [
            TranscriptionFrame(text="你好", user_id="user", timestamp="now"),
            TranscriptionFrame(text="什么是大语言模型", user_id="user", timestamp="now"),
            TranscriptionFrame(text="怎么学习", user_id="user", timestamp="now"),
            TranscriptionFrame(text="简单介绍下transformer原理", user_id="user", timestamp="now"),
            TranscriptionFrame(text="什么是神经元", user_id="user", timestamp="now"),
        ]
        test_frames = []
        for item in text_frames:
            test_frames.append(UserStartedSpeakingFrame())
            test_frames.append(item)
            test_frames.append(UserStoppedSpeakingFrame())
        test_frames.append(StopTaskFrame())
        await task.queue_frames(test_frames)
        runner = PipelineRunner()
        await runner.run(task)

        logging.info(f"answer: {mock_proc.answer}")

        self.assertGreater(len(mock_proc.token), 0)
        self.assertEqual(len(mock_proc.answer), len(text_frames))

        logging.info(f"all msgs: {self.message_store}")
