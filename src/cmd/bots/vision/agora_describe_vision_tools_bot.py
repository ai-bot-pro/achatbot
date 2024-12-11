import logging
import os
from typing import Any, Awaitable, Callable

from apipeline.pipeline.pipeline import Pipeline, FrameDirection
from apipeline.pipeline.runner import PipelineRunner
from apipeline.pipeline.task import PipelineTask, PipelineParams

from src.processors.aggregators.openai_llm_context import (
    OpenAIAssistantContextAggregator,
    OpenAILLMContext,
    OpenAIUserContextAggregator,
)
from src.processors.llm.base import LLMProcessor
from src.processors.user_image_request_processor import UserImageOrTextRequestProcessor
from src.processors.aggregators.vision_image_frame import VisionImageFrameAggregator
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.common.types import AgoraParams
from src.cmd.bots.base_agora import AgoraChannelBot
from src.transports.agora import AgoraTransport
from src.types.frames.control_frames import UserImageRequestFrame
from .. import register_ai_room_bots


@register_ai_room_bots.register
class AgoraDescribeVisionToolsBot(AgoraChannelBot):
    r"""
    - use init user image request promts to get image to vision llm
    - if not in image request promts, need use system instruct promt (zero/few shot)
        to run vision tools to get image to vision llm
    """

    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.max_function_call_cn = int(os.environ.get("MAX_FUNCTION_CALL_CN", "3"))
        self.init_bot_config()

    async def get_weather(
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

        location = arguments["location"]
        # just a mock response
        # add result to assistant context
        self.get_weather_call_cn += 1
        if self.max_function_call_cn > self.get_weather_call_cn:
            await result_callback(f"The weather in {location} is currently 72 degrees and sunny.")
        else:
            self.get_weather_call_cn = 0

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

        self.describe_image_cn += 1
        if self.max_function_call_cn > self.get_weather_call_cn:
            # await result_callback(f"describe image.")
            await llm.push_frame(
                UserImageRequestFrame(self.participant_id), FrameDirection.UPSTREAM
            )
        else:
            self.describe_image_cn = 0

    async def arun(self):
        vad_analyzer = self.get_vad_analyzer()
        agora_params = AgoraParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=vad_analyzer,
            vad_audio_passthrough=True,
            camera_in_enabled=True,
        )

        asr_processor = self.get_asr_processor()

        tts_processor: TTSProcessor = self.get_tts_processor()
        stream_info = tts_processor.get_stream_info()

        agora_params.audio_out_sample_rate = stream_info["sample_rate"]
        agora_params.audio_out_channels = stream_info["channels"]
        transport = AgoraTransport(
            self.args.token,
            params=agora_params,
        )

        init_user_prompts = []
        if self._bot_config.extends and "init_image_request_prompts" in self._bot_config.extends:
            init_user_prompts = self._bot_config.extends["init_image_request_prompts"]
        image_requester = UserImageOrTextRequestProcessor(init_user_prompts=init_user_prompts)
        vision_aggregator = VisionImageFrameAggregator(pass_text=True)

        self.llm_context = OpenAILLMContext()
        if self._bot_config.llm and self._bot_config.llm.messages:
            self.llm_context.set_messages(self._bot_config.llm.messages)
        if self._bot_config.llm and self._bot_config.llm.tools:
            self.llm_context.set_tools(self._bot_config.llm.tools)
        llm_user_ctx_aggr = OpenAIUserContextAggregator(self.llm_context)
        llm_assistant_ctx_aggr = OpenAIAssistantContextAggregator(llm_user_ctx_aggr)

        llm_processor = self.get_remote_llm_processor()
        llm_processor.register_function("get_weather", self.get_weather)
        llm_processor.register_function("describe_image", self.describe_image)
        self.get_weather_call_cn = 0
        self.describe_image_cn = 0
        self.participant_id = 0

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(
            transport: AgoraTransport,
            user_id: str,
        ):
            # subscribed the first participant
            transport.capture_participant_video(user_id, framerate=0)
            image_requester.set_participant_id(user_id)
            self.participant_id = user_id

            participant_name = user_id
            await tts_processor.say(
                f"你好,{participant_name}。"
                f"欢迎使用 Vision Tools Bot."
                f"我是一名虚拟工具型助手，可以结合视频进行提问。"
            )

        pipeline = Pipeline(
            [
                transport.input_processor(),
                asr_processor,
                image_requester,
                vision_aggregator,
                llm_user_ctx_aggr,
                llm_processor,
                tts_processor,
                transport.output_processor(),
                llm_assistant_ctx_aggr,
            ]
        )
        self.task = PipelineTask(pipeline, params=PipelineParams())
        await PipelineRunner().run(self.task)
