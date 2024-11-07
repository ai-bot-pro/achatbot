import logging

from apipeline.frames import Frame
from apipeline.processors.filters.null_filter import NullFilter
from apipeline.processors.filters.function_filter import FunctionFilter
from apipeline.notifiers.event_notifier import EventNotifier
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.parallel_pipeline import ParallelPipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.processors.aggregators.hold import HoldFramesAggregator, HoldLastFrameAggregator

from src.processors.user_idle_processor import UserIdleProcessor
from src.processors.aggregators.llm_response import LLMAssistantResponseAggregator, LLMUserResponseAggregator
from src.processors.llm.base import LLMProcessor
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from src.common.types import DailyParams
from src.transports.daily import DailyTransport
from src.cmd.bots.base_daily import DailyRoomBot
from src.cmd.bots import register_ai_room_bots
from src.types.frames import LLMMessagesFrame, TextFrame

from dotenv import load_dotenv

load_dotenv(override=True)


@register_ai_room_bots.register
class DailyNaturalConversationBot(DailyRoomBot):
    """
    natural conversation bot with daily webrtc
    use 2 llm to conversation with user
    - statement_llm_processor: use llm to do sentence completion NLP task
    - chat_llm_processor: chat
    """

    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()
        self._notifier = EventNotifier()

    async def wake_notifier_filter(self, frame: Frame):
        if isinstance(frame, TextFrame) and frame.text == "YES":
            await self._notifier.notify()
        return True

    async def arun(self):
        vad_analyzer = VADAnalyzerEnvInit.initVADAnalyzerEngine()
        self.daily_params = DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=vad_analyzer,
            vad_audio_passthrough=True,
            transcription_enabled=False,
        )

        asr_processor = self.get_asr_processor()

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

        chat_llm_processor: LLMProcessor = self.get_llm_processor()
        chat_messages = [{
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
        }]
        if self._bot_config.llm.messages:
            chat_messages = self._bot_config.llm.messages
        user_response = LLMUserResponseAggregator(chat_messages)
        assistant_response = LLMAssistantResponseAggregator(chat_messages)

        statement_llm_processor: LLMProcessor = self.get_llm_processor()
        statement_messages = [{
            "role": "system",
            "content": "Determine if the user's statement is a complete sentence or question, ending in a natural pause or punctuation. Return 'YES' if it is complete and 'NO' if it seems to leave a thought unfinished.",
        }]
        if self._bot_config.nlp_task_llm.messages:
            statement_messages = self._bot_config.nlp_task_llm.messages
        statement_user_response = LLMUserResponseAggregator(statement_messages)

        # Sometimes the LLM will fail detecting if a user has completed a sentence,
        # this will wake up the notifier if that happens.
        async def idle_callback(processor: UserIdleProcessor):
            await self._notifier.notify()
        user_idle = UserIdleProcessor(callback=idle_callback, timeout=3.0)

        # This processor keeps the last context
        # and will let it through once the notifier is woken up.
        hold_last_frame_aggregator = HoldLastFrameAggregator(
            self._notifier, hold_frame_classes=(LLMMessagesFrame,))

        self.task = PipelineTask(
            Pipeline([
                transport.input_processor(),
                asr_processor,
                ParallelPipeline(
                    [
                        statement_user_response,
                        statement_llm_processor,
                        FunctionFilter(filter=self.wake_notifier_filter),
                        NullFilter(),
                    ],
                    [
                        user_response,
                        hold_last_frame_aggregator,
                        chat_llm_processor,
                    ],
                ),
                user_idle,
                tts_processor,
                transport.output_processor(),
                assistant_response,
            ]),
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                send_initial_empty_metrics=False,
            ),
        )

        transport.add_event_handlers(
            "on_first_participant_joined",
            [self.on_first_participant_joined, self.on_first_participant_say_hi])
        transport.add_event_handler(
            "on_participant_left",
            self.on_participant_left)
        transport.add_event_handler(
            "on_call_state_updated",
            self.on_call_state_updated)

        await PipelineRunner().run(self.task)

    async def on_first_participant_say_hi(self, transport: DailyTransport, participant):
        self.session.set_client_id(participant["id"])
        if self.daily_params.transcription_enabled:
            transport.capture_participant_transcription(participant["id"])

        # joined use tts say "hello" to introduce with llm generate
        if self._bot_config.tts \
                and self._bot_config.llm \
                and self._bot_config.llm.messages \
                and len(self._bot_config.llm.messages) == 1:
            hi_text = "Please introduce yourself first."
            if self._bot_config.llm.language \
                    and self._bot_config.llm.language == "zh":
                hi_text = "请用中文介绍下自己。"
            self._bot_config.llm.messages.append({
                "role": "user",
                "content": hi_text,
            })
            await self.task.queue_frames([LLMMessagesFrame(self._bot_config.llm.messages)])
