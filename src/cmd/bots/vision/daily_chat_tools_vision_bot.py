import asyncio
import logging
import os
from typing import Any, Awaitable, Callable

from PIL import Image

from apipeline.frames.data_frames import TextFrame
from apipeline.frames.control_frames import EndFrame
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.runner import PipelineRunner
from apipeline.pipeline.task import PipelineTask, PipelineParams
from apipeline.processors.aggregators.sentence import SentenceAggregator
from apipeline.processors.output_processor import OutputFrameProcessor
from apipeline.processors.logger import FrameLogger

from src.processors.image_capture_processor import ImageCaptureProcessor
from src.processors.aggregators.vision_image_frame import VisionImageFrameAggregator
from src.processors.llm.base import LLMProcessor
from src.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAIAssistantContextAggregator,
    OpenAIUserContextAggregator,
)
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.common.types import DailyParams
from src.cmd.bots.base_daily import DailyRoomBot
from src.transports.daily import DailyTransport
from src.types.frames.data_frames import (
    FunctionCallResultFrame,
    LLMMessagesFrame,
    UserImageRawFrame,
    VisionImageRawFrame,
)
from src.common.register import Register
from .. import register_ai_room_bots

register_tool_funtions = Register("daily-chat-vision-tool-functions")


@register_ai_room_bots.register
class DailyChatToolsVisionBot(DailyRoomBot):
    r"""
    use function tools llm model to chat (for text LLM)
    - when tool is describe image, use describe_image function which use vision model to describe the image with describe text (vision LLM)
        - just use vision lm(Vertical Domain(OCR, etc..) or General(VIT)) to detecte objects and describe the image/video with text
    - when tool is get weather, use get_weather function to get weahter info. this function just a demo

    - TODO: do more tool functions, e.g.: music gen/search, google search etc... become a agent
    !NOTE: need write system prompt to guide LLM to answer

    !THINKING: @weedge (gedigedix3) personal AI(agent)
    - if want a system engine become more easy, need train a MLLM(which can call tools with post training) to supporte e2e text,audio,vision (a big bang!);
    - if not, need base text LLM(which can call tools with post training) and more engine pipeline with tools(agents) to support ~!
    - if just develop app with using lm; prompt is all your need (zero or few shots), need know MLLM/LLM can do what, follow lm capability.
    - all tool function need to async run and set task timeout with using LLM's Parallel Tool
    - multi async agent sync msg with connector(local pipe; async queue or sync rpc)
    — Think about logically rigorous scenarios:
        - like o1 (need CoT prompting with llm, or use openai o1.)
    """

    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.participant_uid = None  # support one user which chats with many bots
        self.max_function_call_cn = int(os.environ.get("MAX_FUNCTION_CALL_CN", "3"))
        self.vision_result = ""
        self.llm_context = OpenAILLMContext()
        self.init_bot_config()

    async def sink_out_cb(self, frame):
        """
        get vision processor result
        """
        if isinstance(frame, TextFrame):
            self.vision_result += frame.text

    @register_tool_funtions.register
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
            await result_callback(f"The weather in {location} is currently 32 degrees and sunny.")
        else:
            self.get_weather_call_cn = 0

    @register_tool_funtions.register
    async def describe_image(
        self,
        function_name: str,
        tool_call_id: str,
        arguments: Any,
        llm: LLMProcessor,
        context: OpenAILLMContext,
        result_callback: Callable[[Any], Awaitable[None]],
    ):
        logging.info(
            f"function_name:{function_name}, tool_call_id:{tool_call_id},"
            f"arguments:{arguments}, llm:{llm}, context:{context}"
        )
        if "question" not in arguments:
            arguments["question"] = "describe image."
        images = self.image_capture_processor.capture_imgs.get(cls=list)
        if len(images) == 0:
            # no described image, so return tips
            await result_callback("no described image, please try again.")
            return

        image: UserImageRawFrame = images[0]
        frame = VisionImageRawFrame(
            text=arguments["question"],
            image=image.image,
            size=image.size,
            format=image.format,
            mode=image.mode,
        )
        vision_task = PipelineTask(
            Pipeline(
                [
                    # SentenceAggregator(),
                    # UserImageRequestProcessor(),
                    # VisionImageFrameAggregator(),
                    self.vision_llm_processor,
                    OutputFrameProcessor(cb=self.sink_out_cb),
                ]
            )
        )
        await vision_task.queue_frames(
            [
                frame,
                EndFrame(),
            ]
        )
        await PipelineRunner().run(vision_task)
        # return describe image tool result
        await result_callback(self.vision_result)

    async def arun(self):
        self.image_capture_processor = ImageCaptureProcessor()
        self.vision_llm_processor = self.get_vision_llm_processor()

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
            self.args.room_url,
            self.args.token,
            self.args.bot_name,
            self.daily_params,
        )
        transport.add_event_handler("on_first_participant_joined", self.on_first_participant_joined)
        transport.add_event_handler("on_participant_left", self.on_participant_left)
        transport.add_event_handler("on_call_state_updated", self.on_call_state_updated)

        if self._bot_config.llm and self._bot_config.llm.messages:
            self.llm_context.set_messages(self._bot_config.llm.messages)
        if self._bot_config.llm and self._bot_config.llm.tools:
            self.llm_context.set_tools(self._bot_config.llm.tools)
        llm_user_ctx_aggr = OpenAIUserContextAggregator(self.llm_context)
        llm_assistant_ctx_aggr = OpenAIAssistantContextAggregator(llm_user_ctx_aggr)
        llm_processor = self.get_openai_llm_processor()

        # register function
        logging.info(f"register tool functions: {register_tool_funtions.items()}")
        llm_processor.register_function("get_weather", self.get_weather)
        self.get_weather_call_cn = 0
        llm_processor.register_function("describe_image", self.describe_image)
        self.describe_image_call_cn = 0

        pipeline = Pipeline(
            [
                transport.input_processor(),
                # FrameLogger(include_frame_types=[UserImageRawFrame]),
                self.image_capture_processor,
                asr_processor,
                llm_user_ctx_aggr,
                llm_processor,
                tts_processor,
                transport.output_processor(),
                # FrameLogger(include_frame_types=[FunctionCallResultFrame]),
                llm_assistant_ctx_aggr,
            ]
        )
        self.task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=False))
        await PipelineRunner().run(self.task)

    async def on_first_participant_joined(self, transport: DailyTransport, participant):
        self.participant_uid = participant["id"]
        transport.capture_participant_video(participant["id"])
        if self.daily_params.transcription_enabled:
            transport.capture_participant_transcription(participant["id"])

        # joined use tts say "hello" to introduce with llm generate
        if (
            self._bot_config.tts
            and self._bot_config.llm
            and self._bot_config.llm.messages
            and len(self._bot_config.llm.messages) == 1
        ):
            hi_text = "Please introduce yourself first."
            if self._bot_config.llm.language and self._bot_config.llm.language == "zh":
                hi_text = "请用中文介绍下自己。"
            self._bot_config.llm.messages.append(
                {
                    "role": "user",
                    "content": hi_text,
                }
            )
            await self.task.queue_frames([LLMMessagesFrame(self._bot_config.llm.messages)])
