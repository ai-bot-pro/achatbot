import os
import logging

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import TiDBVectorStore

from langchain_community.embeddings import JinaEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.processors.frame_processor import FrameProcessor

from src.common.factory import EngineClass
from src.modules.speech.tts import TTSEnvInit
from src.modules.speech.asr import ASREnvInit
from src.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.common import interface
from src.processors.aggregators.llm_response import LLMAssistantResponseAggregator, LLMUserResponseAggregator
from src.processors.ai_frameworks.langchain_rag_processor import LangchainRAGProcessor
from src.processors.speech.audio_volume_time_processor import AudioVolumeTimeProcessor
from src.processors.speech.asr.base import TranscriptionTimingLogProcessor
from src.processors.rtvi_processor import RTVIASRConfig, RTVIConfig, RTVILLMConfig, RTVITTSConfig
from src.processors.speech.asr.asr_processor import AsrProcessor
from src.common.types import DailyParams
from src.transports.daily import DailyTransport
from src.cmd.bots.rag.helper import get_tidb_url
from src.cmd.bots.base import DailyRoomBot, register_daily_room_bots

from dotenv import load_dotenv
load_dotenv(override=True)

DEFAULT_SYSTEM_PROMPT = """
You are Andrej Karpathy, a Slovak-Canadian computer scientist who served as the director of artificial intelligence and Autopilot Vision at Tesla. \
    You co-founded and formerly worked at OpenAI, where you specialized in deep learning and computer vision. \
    You publish Youtube videos in which you explain complex machine learning concepts. \
    Your job is to help people with the content in your Youtube videos given context . \
    Keep your responses concise and relatively simple. \
    Ask for clarification if a user question is ambiguous. Be nice and helpful. Ensure responses contain only words. \
    Check again that you have not included special characters other than '?' or '!'. \
    Do not output in markdown format.
"""


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
        r"""
        reuse rtvi bot config
        !TODONE: need config processor with bot config (redefine api params) @weedge
        bot config: Dict[str, Dict[str,Any]]
        e.g. {"llm":{"key":val,"tag":TAG,"args":{}}, "tts":{"key":val,"tag":TAG,"args":{}}}
        """
        super().__init__(**args)
        self.message_store = {}
        try:
            logging.debug(f'config: {self.args.bot_config}')
            self._bot_config: RTVIConfig = RTVIConfig(**self.args.bot_config)
            if self._bot_config.llm is None:
                self._bot_config.llm = RTVILLMConfig()
        except Exception as e:
            raise Exception("Failed to parse bot configuration")

    def bot_config(self):
        return self._bot_config.model_dump()

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.message_store:
            self.message_store[session_id] = ChatMessageHistory()
        return self.message_store[session_id]

    async def arun(self):
        vad_analyzer = VADAnalyzerEnvInit.initVADAnalyzerEngine()
        daily_params = DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=vad_analyzer,
            vad_audio_passthrough=True,
            transcription_enabled=False,
        )

        # !NOTE: u can config env in .env file to init default
        # or api config
        asr_processor: FrameProcessor = None
        if self._bot_config.asr \
                and self._bot_config.asr.tag == "deepgram_asr_processor" \
                and self._bot_config.asr.args:
            from src.processors.speech.asr.deepgram_asr_processor import DeepgramAsrProcessor
            asr_processor = DeepgramAsrProcessor(
                api_key=os.getenv("DEEPGRAM_API_KEY"),
                **self._bot_config.asr.args)
        else:
            # use asr engine processor
            asr: interface.IAsr | EngineClass | None = None
            if self._bot_config.asr \
                    and self._bot_config.asr.tag \
                    and self._bot_config.asr.args:
                asr = ASREnvInit.getEngine(
                    self._bot_config.asr.tag, **self._bot_config.asr.args)
            else:
                logging.info(f"use default asr engine processor")
                asr = ASREnvInit.initASREngine()
                self._bot_config.asr = RTVIASRConfig(tag=asr.SELECTED_TAG, args=asr.get_args_dict())
            asr_processor = AsrProcessor(
                asr=asr,
                session=self.session
            )

        tts_processor: FrameProcessor | None = None

        if self._bot_config.tts and self._bot_config.tts.tag == "elevenlabs_tts_processor":
            from src.processors.speech.tts.elevenlabs_tts_processor import ElevenLabsTTSProcessor
            tts_processor = ElevenLabsTTSProcessor(**self._bot_config.tts.args)
        elif self._bot_config.tts and self._bot_config.tts.tag == "cartesia_tts_processor":
            from src.processors.speech.tts.cartesia_tts_processor import CartesiaTTSProcessor
            tts_processor = CartesiaTTSProcessor(
                # voice_id=self._bot_config.tts.voice,
                # cartesia_version="2024-06-10",
                # model_id="sonic-multilingual",
                # language=self._bot_config.tts.language if self._bot_config.tts.language else "en",
                **self._bot_config.tts.args
            )
        else:
            # use tts engine processor
            tts: interface.ITts | EngineClass | None = None
            if self._bot_config.tts \
                    and self._bot_config.tts.tag \
                    and self._bot_config.tts.args:
                tts = TTSEnvInit.getEngine(
                    self._bot_config.tts.tag, **self._bot_config.tts.args)
            else:
                # default tts engine processor
                logging.info(f"use default tts engine processor")
                tts = TTSEnvInit.initTTSEngine()
                self._bot_config.tts = RTVITTSConfig(tag=tts.SELECTED_TAG, args=tts.get_args_dict())

            stream_info = tts.get_stream_info()
            daily_params.audio_out_sample_rate = stream_info["rate"]
            daily_params.audio_out_channels = stream_info["channels"]
            tts_processor = TTSProcessor(tts=tts, session=self.session)

        # default use openai llm processor
        api_key = os.environ.get("OPENAI_API_KEY")
        if "groq" in self._bot_config.llm.base_url:
            api_key = os.environ.get("GROQ_API_KEY")
        elif "together" in self._bot_config.llm.base_url:
            api_key = os.environ.get("TOGETHER_API_KEY")
        llm = ChatOpenAI(
            model=self._bot_config.llm.model,
            base_url=self._bot_config.llm.base_url,
            api_key=api_key,
            temperature=0.8,
            top_p=0.7,
            max_retries=3,
            max_tokens=1024,
        )

        model_name = "jina-embeddings-v2-base-en"
        if self._bot_config.llm.language == "zh":
            model_name = "jina-embeddings-v2-base-zh"
        vectorstore = TiDBVectorStore(
            connection_string=get_tidb_url(),
            embedding_function=JinaEmbeddings(
                jina_api_key=os.getenv('JINA_API_KEY'),
                model_name=model_name,
            ),
            table_name="AndrejKarpathy",
            distance_strategy=os.getenv('TIDB_VSS_DISTANCE_STRATEGY', 'cosine'),
        )
        score_threshold = 0.8
        if self._bot_config.llm.language == "zh":
            score_threshold = 0.5
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": score_threshold},
        )

        system_prompt = DEFAULT_SYSTEM_PROMPT
        if self._bot_config.llm.language == "zh":
            system_prompt += " Please communicate in Chinese"
        if self._bot_config.llm.messages and len(self._bot_config.llm.messages) > 0 \
                and len(self._bot_config.llm.messages[0]['content']) > 0:
            system_prompt = self._bot_config.llm.messages[0]['content']
        logging.info(f"use system prompt: {system_prompt}")
        answer_prompt = ChatPromptTemplate.from_messages([(
            "system", system_prompt + """ \
                {context}"""), MessagesPlaceholder("chat_history"), ("human", "{input}"), ])
        question_answer_chain = create_stuff_documents_chain(llm, answer_prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        # chain = prompt | rag
        rag_history_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            history_messages_key="chat_history",
            input_messages_key="input",
            output_messages_key="answer")
        self.langchain_processor = LangchainRAGProcessor(chain=rag_history_chain)

        avt_processor = AudioVolumeTimeProcessor()
        tl_processor = TranscriptionTimingLogProcessor(avt_processor)

        llm_in_aggr = LLMUserResponseAggregator()
        llm_out_aggr = LLMAssistantResponseAggregator()

        transport = DailyTransport(
            self.args.room_url,
            self.args.token,
            self.args.bot_name,
            daily_params,
        )

        pipeline = Pipeline([
            transport.input_processor(),   # Transport user input
            avt_processor,  # Audio volume timer
            asr_processor,  # Speech-to-text
            tl_processor,  # Transcription timing logger
            llm_in_aggr,              # User responses
            self.langchain_processor,  # RAG
            tts_processor,       # TTS
            transport.output_processor(),  # Transport bot output
            llm_out_aggr,             # Assistant spoken responses
        ])

        task = PipelineTask(pipeline, PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            report_only_initial_ttfb=True,
        ))

        transport.add_event_handler(
            "on_first_participant_joined",
            self.on_first_participant_joined)
        transport.add_event_handler(
            "on_participant_left",
            self.on_participant_left)
        transport.add_event_handler(
            "on_call_state_updated",
            self.on_call_state_updated)

        runner = PipelineRunner()

        await runner.run(task)

    async def on_first_participant_joined(self, transport: DailyTransport, participant):
        self.session.set_client_id(participant['id'])
        self.langchain_processor.set_participant_id(participant['id'])
        transport.capture_participant_transcription(participant["id"])
        logging.info(f"First participant {participant['id']} joined")
