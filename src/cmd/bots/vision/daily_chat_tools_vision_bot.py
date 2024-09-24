import asyncio
import logging
from typing import Any, Awaitable, Callable

from PIL import Image

from apipeline.frames.data_frames import TextFrame
from apipeline.frames.control_frames import EndFrame
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.runner import PipelineRunner
from apipeline.pipeline.task import PipelineTask, PipelineParams
from apipeline.processors.aggregators.sentence import SentenceAggregator
from apipeline.processors.output_processor import OutputFrameProcessor

from src.processors.aggregators.vision_image_frame import VisionImageFrameAggregator
from src.processors.user_image_request_processor import UserImageRequestProcessor
from src.processors.llm.base import LLMProcessor
from src.processors.aggregators.llm_response import OpenAIAssistantContextAggregator, OpenAIUserContextAggregator
from src.processors.aggregators.openai_llm_context import OpenAILLMContext
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.common.types import DailyParams
from src.cmd.bots.base import DailyRoomBot
from src.transports.daily import DailyTransport
from src.types.frames.data_frames import LLMMessagesFrame, UserImageRawFrame, VisionImageRawFrame
from src.common.register import Register
from .. import register_daily_room_bots

register_tool_funtions = Register('daily-chat-vision-tool-functions')


@register_daily_room_bots.register
class DailyChatToolsVisionBot(DailyRoomBot):
    r"""
    use function tools llm model to chat (text LLM)
    - when tool is describe image, use describe_image function which use vision model to describe the image with describe text (vision LLM)
        - just use vision lm(Vertical Domain(OCR, etc..) or General(VIT)) to detecte objects and describe the image/video with text
    - when tool is get weather, use get_weather function to get weahter info. this function just a demo

    - TODO: do more tool functions, e.g.: music gen/search, google search etc... become a agent
    !NOTE: need write system prompt to guide LLM to answer

    !THINKING: @weedge (gedigedix3) personal AI(agent)
    - if want a system engine become more easy, need train a LWM(which can call tools with post training) to supporte e2e text,audio,vision (a big bang!);
    - if not, need base text LLM(which can call tools with post training) and more engine pipeline with tools(agents) to support ~!
    - if just develop app with using lm; prompt is all your need (zero or few shots), need know LWM/LLM can do what, follow lm capability.
    """

    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.vision_task = None
        self.participant_uid = None  # support one user which chats with many bots
        self.vision_result = ""
        self.llm_context = OpenAILLMContext()
        self.init_bot_config()

    async def sink_out_cb(self, frame):
        """
        get vision processor result
        """
        if isinstance(frame, TextFrame):
            self.vision_result += frame.text

    async def init_vision_pipeline_task(self):
        out_processor = OutputFrameProcessor(cb=self.sink_out_cb)
        pipeline = Pipeline([
            # SentenceAggregator(),
            # UserImageRequestProcessor(),
            # VisionImageFrameAggregator(),
            self.get_vision_llm_processor(),
            out_processor,
        ])
        self.vision_task = PipelineTask(pipeline, params=PipelineParams())

    @register_tool_funtions.register
    async def get_weather(
            self,
            function_name: str,
            tool_call_id: str,
            arguments: Any,
            llm: LLMProcessor,
            context: OpenAILLMContext,
            result_callback: Callable[[Any], Awaitable[None]]):
        location = arguments["location"]
        logging.info(
            f"function_name:{function_name}, tool_call_id:{tool_call_id},"
            f"arguments:{arguments}, llm:{llm}, context:{context}")
        # just a mock response
        # add result to assistant context
        await result_callback(f"The weather in {location} is currently 72 degrees and sunny.")

    @register_tool_funtions.register
    async def get_image(
            self,
            function_name: str,
            tool_call_id: str,
            arguments: Any,
            llm: LLMProcessor,
            context: OpenAILLMContext,
            result_callback: Callable[[Any], Awaitable[None]]):
        logging.info(
            f"function_name:{function_name}, tool_call_id:{tool_call_id},"
            f"arguments:{arguments}, llm:{llm}, context:{context}")
        if "question" in arguments:
            frame = VisionImageRawFrame(
                text=arguments["question"],
                image=bytes([]),
                size=(0, 0),
                format=None,
                mode=None,
            )
            if "image" in arguments \
                    and isinstance(arguments["image"], UserImageRawFrame) \
                    and arguments["image"].user_id == self.participant_uid:
                frame.image = arguments["image"].image
                frame.size = arguments["image"].size
                frame.format = arguments["image"].format
                frame.mode = arguments["image"].mode
            await self.vision_task.queue_frames([
                frame,
                EndFrame(),
            ])
            await PipelineRunner().run(self.vision_task)
            await result_callback(self.vision_result)

    async def arun(self):
        await self.init_vision_pipeline_task()

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

        tts_processor: TTSProcessor = self.get_tts_processor()
        stream_info = tts_processor.get_stream_info()

        self.daily_params.audio_out_sample_rate = stream_info["sample_rate"]
        self.daily_params.audio_out_channels = stream_info["channels"]
        transport = DailyTransport(
            self.args.room_url, self.args.token, self.args.bot_name,
            self.daily_params,
        )
        transport.add_event_handler(
            "on_first_participant_joined",
            self.on_first_participant_joined)
        transport.add_event_handler(
            "on_participant_left",
            self.on_participant_left)
        transport.add_event_handler(
            "on_call_state_updated",
            self.on_call_state_updated)

        if self._bot_config.llm and self._bot_config.llm.messages:
            self.llm_context.add_messages(self._bot_config.llm.messages)
        if self._bot_config.llm and self._bot_config.llm.tools:
            self.llm_context.set_tools(self._bot_config.llm.tools)
        llm_user_ctx_aggr = OpenAIUserContextAggregator(self.llm_context)
        llm_assistant_ctx_aggr = OpenAIAssistantContextAggregator(llm_user_ctx_aggr)
        llm_processor = self.get_openai_llm_processor()

        # register function
        logging.info(f"register tool functions: {register_tool_funtions.items()}")
        llm_processor.register_function("get_weather", self.get_weather)
        llm_processor.register_function("get_image", self.get_image)

        pipeline = Pipeline([
            transport.input_processor(),
            asr_processor,
            llm_user_ctx_aggr,
            llm_processor,
            tts_processor,
            transport.output_processor(),
            llm_assistant_ctx_aggr,
        ])
        self.task = PipelineTask(pipeline, params=PipelineParams())
        await PipelineRunner().run(self.task)

    async def on_first_participant_joined(self, transport: DailyTransport, participant):
        self.participant_uid = participant["id"]
        transport.capture_participant_video(participant["id"], framerate=0)
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
