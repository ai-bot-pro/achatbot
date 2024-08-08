import os
import logging
import asyncio

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import TiDBVectorStore


from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner

from src.processors.llm.openai_llm_processor import OpenAILLMProcessor
from src.processors.speech.tts.cartesia_tts_processor import CartesiaTTSProcessor
from src.processors.rtvi_processor import RTVIConfig, RTVIProcessor, RTVISetup
from src.processors.speech.asr.asr_processor import AsrProcessor
from src.modules.speech.vad_analyzer.silero import SileroVADAnalyzer
from src.common.types import DailyParams, DailyRoomBotArgs, DailyTranscriptionSettings
from src.transports.daily import DailyTransport
from src.cmd.init import Env
from ..base import DailyRoomBot, register_daily_room_bots

from dotenv import load_dotenv
load_dotenv(override=True)


@register_daily_room_bots.register
class DailyLangchainRAGBot(DailyRoomBot):
    """
    Video playback scenario(e.g. education/meeting video playback).
    offline pipeline:
    need video speech transcript to text
    embedding text stores in vector db
    (increamental update for semantic text search,
    next maybe for multimoding data)

    online realtime pipeline:
    asr gen text
    retrieve text from vector db
    gen prompt from retrived text
    llm gen repsonse text
    tts gen audio
    """

    def __init__(self, **args) -> None:
        super().__init__(**args)
        try:
            logging.debug(f'config: {self.args.bot_config}')
            self._bot_config: RTVIConfig = RTVIConfig(**self.args.bot_config)
        except Exception as e:
            raise Exception("Failed to parse bot configuration")

    def bot_config(self):
        return self._bot_config.model_dump()

    async def _run(self):
        transport = DailyTransport(
            self.args.room_url,
            self.args.token,
            self.args.bot_name,
            DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                vad_audio_passthrough=True,
                transcription_enabled=False,
            ))

        # !NOTE: u can config env in .env file
        asr_processor = AsrProcessor(
            asr=Env.initASREngine(),
            session=self.session
        )

        # https://docs.cartesia.ai/getting-started/available-models
        # !NOTE: Timestamps are not supported for language 'zh'
        tts_processor = CartesiaTTSProcessor(
            voice_id=self._bot_config.tts.voice,
            cartesia_version="2024-06-10",
            model_id="sonic-multilingual",
            language="zh",
        )

        # now just reuse rtvi bot config
        # !TODO: need config processor with bot config (redefine api params) @weedge
        # bot config: Dict[str, Dict[str,Any]]
        # e.g. {"llm":{"key":val,"tag":TAG}, "tts":{"key":val,"tag":TAG}}
        llm = ChatOpenAI(
            model=self._bot_config.llm.model,
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        )
        vectorstore = PineconeVectorStore.from_existing_index(
            "andrej-youtube", OpenAIEmbeddings()
        )
        retriever = vectorstore.as_retriever()

        answer_prompt = ChatPromptTemplate.from_messages([(
            "system", """You are Andrej Karpathy, a Slovak-Canadian computer scientist who served as the director of artificial intelligence and Autopilot Vision at Tesla. \
                 You co-founded and formerly worked at OpenAI, where you specialized in deep learning and computer vision. You publish Youtube videos in which you explain complex \
                 machine learning concepts. Your job is to help people with the content in your Youtube videos given context . Keep your responses concise and relatively simple. \
                Ask for clarification if a user question is ambiguous. Be nice and helpful. Ensure responses contain only words. Check again that you have not included special characters other than '?' or '!'. \

                {context}"""), MessagesPlaceholder("chat_history"), ("human", "{input}"), ])
        question_answer_chain = create_stuff_documents_chain(llm, answer_prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # chain = prompt | llm
        history_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            history_messages_key="chat_history",
            input_messages_key="input",
            output_messages_key="answer")
        lc = LangchainRAGProcessor(chain=history_chain)
