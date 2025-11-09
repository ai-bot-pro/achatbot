import os
import logging
from typing import Union

from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.processors.logger import FrameLogger

from src.processors.llm.base import LLMProcessor
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.common.types import DailyParams
from src.transports.daily import DailyTransport
from src.cmd.bots.base_daily import DailyRoomBot
from src.cmd.bots import register_ai_room_bots
from src.types.frames.data_frames import LLMMessagesFrame, TextFrame
from src.types.frames.control_frames import (
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
)
from src.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAIAssistantContextAggregator,
    OpenAIUserContextAggregator,
)

from dotenv import load_dotenv

load_dotenv(override=True)


@register_ai_room_bots.register
class DailyLangchainMemBot(DailyRoomBot):
    """
    use asr processor, don't use daily transcirption
    """

    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()

    def get_langchain_memory_processor(self):
        from langchain_openai import ChatOpenAI
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import InMemoryVectorStore, Chroma

        # zhipu GLM
        llm = ChatOpenAI(
            base_url="https://open.bigmodel.cn/api/paas/v4",
            model="glm-4.5-flash",
            max_tokens=32768,
            temperature=0.2,
            api_key=os.getenv("ZHIPU_API_KEY"),
        )

        # Set embeddings
        # https://ai.gitee.com/serverless-api#embedding-rerank
        # 100 api calls per day, free tier
        embd = OpenAIEmbeddings(
            base_url="https://ai.gitee.com/v1",
            model="Qwen3-Embedding-8B",  # 4096
            api_key=os.environ["GITEE_API_KEY"],
            dimensions=1536,
            check_embedding_ctx_length=False,
            chunk_size=1000,
        )
        # vector_store = InMemoryVectorStore(embedding=embd)

        # RAG
        config = {
            "llm": {"provider": "langchain", "config": {"model": llm}},
            "embedder": {"provider": "langchain", "config": {"model": embd}},
            # "vector_store": {
            #   "provider": "langchain",
            #   "config": {"client": vector_store},
            # },  # default qdrant
            # "graph_store": None,
            # "ranker": None,
        }
        memory_processor = self.get_memory_processor(local_config=config)
        return memory_processor

    async def arun(self):
        vad_analyzer = self.get_vad_analyzer()
        self.daily_params = DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=vad_analyzer,
            vad_audio_passthrough=True,
            transcription_enabled=False,
        )

        asr_processor = self.get_asr_processor()

        llm_processor: LLMProcessor = self.get_llm_processor()

        tts_processor: TTSProcessor = self.get_tts_processor()
        stream_info = tts_processor.get_stream_info()
        self.daily_params.audio_out_sample_rate = stream_info["sample_rate"]
        self.daily_params.audio_out_channels = stream_info["channels"]

        transport = DailyTransport(
            self.args.room_url,
            self.args.token,
            self.args.bot_name,
            self.daily_params,
        )

        self.llm_context = OpenAILLMContext()
        messages = []
        if self._bot_config.llm.messages:
            messages = self._bot_config.llm.messages
            self.llm_context.set_messages(messages)
        llm_user_ctx_aggr = OpenAIUserContextAggregator(self.llm_context)
        llm_assistant_ctx_aggr = OpenAIAssistantContextAggregator(llm_user_ctx_aggr)

        self.memory_processor = self.get_langchain_memory_processor()

        self.task = PipelineTask(
            Pipeline(
                [
                    transport.input_processor(),
                    FrameLogger(
                        include_frame_types=[UserStartedSpeakingFrame, UserStoppedSpeakingFrame]
                    ),
                    asr_processor,
                    llm_user_ctx_aggr,
                    self.memory_processor,
                    llm_processor,
                    tts_processor,
                    transport.output_processor(),
                    FrameLogger(
                        include_frame_types=[
                            LLMFullResponseStartFrame,
                            LLMFullResponseEndFrame,
                            # TextFrame,
                        ]
                    ),
                    llm_assistant_ctx_aggr,
                ]
            ),
            params=PipelineParams(
                allow_interruptions=False,
                enable_metrics=True,
                send_initial_empty_metrics=False,
            ),
        )

        transport.add_event_handlers(
            "on_first_participant_joined",
            [self.on_first_participant_joined, self.on_first_participant_say_hi],
        )
        transport.add_event_handler("on_participant_left", self.on_participant_left)
        transport.add_event_handler("on_call_state_updated", self.on_call_state_updated)

        await PipelineRunner().run(self.task)

    async def on_first_participant_say_hi(self, transport: DailyTransport, participant):
        if self.daily_params.transcription_enabled:
            transport.capture_participant_transcription(participant["id"])

        is_cn = (
            self._bot_config.llm
            and self._bot_config.llm.language
            and self._bot_config.llm.language == "zh"
        )

        user_id = participant.get("name") or participant.get("id")
        self.memory_processor.set_user_id(user_id)
        memory_str = await self.memory_processor.get_initial_memories(user_id=user_id)
        if memory_str:
            assistant_greeting = (
                f"基于我们之前的对话，我记得：\n{memory_str}"
                if is_cn
                else f"Based on our previous conversations, I remember:\n{memory_str}"
            )
            self.llm_context.add_message(
                {
                    "role": "assistant",
                    "content": assistant_greeting,
                }
            )

        user_hi_text = "请用中文介绍下自己。" if is_cn else "Please introduce yourself first."
        self.llm_context.add_message(
            {
                "role": "user",
                "content": user_hi_text,
            }
        )

        await self.task.queue_frames([LLMMessagesFrame(self.llm_context.get_messages())])
