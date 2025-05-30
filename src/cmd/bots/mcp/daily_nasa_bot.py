import logging
import os
import shutil

import aiohttp
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.processors.logger import FrameLogger
from apipeline.frames.data_frames import ImageRawFrame
from mcp import StdioServerParameters

from src.processors.image_url_extract_processor import UrlToImageProcessor
from src.processors.llm.base import LLMProcessor
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from src.common.types import DailyParams
from src.transports.daily import DailyTransport
from src.cmd.bots.base_daily import DailyRoomBot
from src.cmd.bots import register_ai_room_bots
from src.types.frames.data_frames import (
    LLMMessagesFrame,
)
from src.services.mcp_client import MCPClient
from src.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAIAssistantContextAggregator,
    OpenAIUserContextAggregator,
)

from dotenv import load_dotenv

load_dotenv(override=True)


@register_ai_room_bots.register
class DailyNASABot(DailyRoomBot):
    """
    daily webrtc + asr + llm(gemini) + nasa mcp server + tts bot
    """

    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()

    async def arun(self):
        vad_analyzer = VADAnalyzerEnvInit.initVADAnalyzerEngine()
        self.daily_params = DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=vad_analyzer,
            vad_audio_passthrough=True,
            transcription_enabled=False,
            camera_out_enabled=True,
            camera_out_width=1024,
            camera_out_height=768,
            # camera_out_is_live=True,
        )

        asr_processor = self.get_asr_processor()

        server_params = StdioServerParameters(
            command=shutil.which("npx"),
            args=["-y", "@programcomputer/nasa-mcp-server@latest"],
            # https://api.nasa.gov
            env={"NASA_API_KEY": os.getenv("NASA_API_KEY")},
        )
        # logging.info(f"{server_params=}")
        mcp_client = MCPClient(server_params=server_params, mcp_name="nasa")

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

        llm_context = OpenAILLMContext()
        # 提问： 给出当天的天文图片； 请解释这张图片的主要内容；我想看2025年5月24日的天文图片;请讲解下图片；
        system = f"""
你是NASA天文探测团队的机器人。
你的目标是以简洁的方式展示你的能力。
你可以使用 NASA MCP 提供的多种工具来帮助用户。
当被要求提供每日天文图片时，请不要向 API 提供任何日期，这将确保我们获取到最新的可用图片。
如果用户要求特定日期的图片，你可以向 API 提供该日期。
你的输出将被转换为音频，所以请不要在回答中包含特殊字符。
以富有创意且有用的方式回应用户所说的内容。
不要过度解释你正在做的事情。
在执行工具调用时，只需用简短的句子回应。
        """
        messages = [{"role": "system", "content": system}]
        if self._bot_config.llm.messages:
            messages = self._bot_config.llm.messages
        llm_context.set_messages(messages)

        tools = await mcp_client.register_tools(llm_processor)
        llm_context.set_tools(tools)

        llm_user_ctx_aggr = OpenAIUserContextAggregator(llm_context)
        llm_assistant_ctx_aggr = OpenAIAssistantContextAggregator(llm_user_ctx_aggr)

        # Create an HTTP session for API calls
        async with aiohttp.ClientSession() as session:
            url2image_processor = UrlToImageProcessor(aiohttp_session=session)
            self.task = PipelineTask(
                Pipeline(
                    [
                        transport.input_processor(),
                        asr_processor,
                        llm_user_ctx_aggr,
                        llm_processor,
                        tts_processor,
                        url2image_processor,  # URL image -> output
                        # FrameLogger(include_frame_types=[ImageRawFrame]),
                        transport.output_processor(),
                        llm_assistant_ctx_aggr,
                    ]
                ),
                params=PipelineParams(
                    allow_interruptions=False,
                    enable_metrics=True,  # deploy prod open it
                    send_initial_empty_metrics=False,
                ),
            )

            transport.add_event_handlers(
                "on_first_participant_joined",
                [self.on_first_participant_joined, self.on_first_participant_say_hi],
            )
            transport.add_event_handler("on_participant_left", self.on_participant_left)
            transport.add_event_handler("on_call_state_updated", self.on_call_state_updated)

            await PipelineRunner(handle_sigint=False).run(self.task)

    async def on_first_participant_say_hi(self, transport: DailyTransport, participant):
        self.session.set_client_id(participant["id"])
        if self.daily_params.transcription_enabled:
            transport.capture_participant_transcription(participant["id"])

        # joined use tts say "hello" to introduce with llm generate
        if (
            self._bot_config.tts
            and self._bot_config.llm
            and self._bot_config.llm.messages is not None
            and isinstance(self._bot_config.llm.messages, list)
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
