import logging

from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.processors.logger import FrameLogger

from src.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAIAssistantContextAggregator,
    OpenAIUserContextAggregator,
)
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.modules.speech.turn_analyzer import TurnAnalyzerEnvInit
from src.common.types import DailyParams
from src.transports.daily import DailyTransport
from src.cmd.bots.base_daily import DailyRoomBot
from src.cmd.bots import register_ai_room_bots
from src.types.frames.data_frames import LLMMessagesFrame
from src.types.frames.control_frames import (
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
)

from dotenv import load_dotenv

load_dotenv(override=True)


@register_ai_room_bots.register
class DailyA2AConversationBot(DailyRoomBot):
    """
    use a2a conversation processor
    """

    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()

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

        self.a2a_processor = self.get_a2a_processor()

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
        if self._bot_config.llm.messages:
            self.llm_context.set_messages(self._bot_config.llm.messages)
        llm_user_ctx_aggr = OpenAIUserContextAggregator(self.llm_context)
        llm_assistant_ctx_aggr = OpenAIAssistantContextAggregator(llm_user_ctx_aggr)

        self.task = PipelineTask(
            Pipeline(
                [
                    transport.input_processor(),
                    FrameLogger(
                        include_frame_types=[UserStartedSpeakingFrame, UserStoppedSpeakingFrame]
                    ),
                    asr_processor,
                    llm_user_ctx_aggr,
                    self.a2a_processor,
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

        await PipelineRunner(handle_sigint=self._handle_sigint).run(self.task)

    async def on_first_participant_say_hi(self, transport: DailyTransport, participant):
        self.a2a_processor.set_user_id(participant["id"])
        await self.a2a_processor.create_conversation()
        if self.daily_params.transcription_enabled:
            transport.capture_participant_transcription(participant["id"])

        is_cn = self._bot_config.a2a and self._bot_config.a2a.language == "zh"

        user_hi_text = "请用中文介绍下自己。" if is_cn else "Please introduce yourself first."
        self.llm_context.add_message(
            {
                "role": "user",
                "content": user_hi_text,
            }
        )

        await self.task.queue_frames([LLMMessagesFrame(self.llm_context.get_messages())])
