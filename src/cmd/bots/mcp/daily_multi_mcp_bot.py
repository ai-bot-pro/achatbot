import logging
import os
import shutil

import aiohttp
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.processors.logger import FrameLogger
from apipeline.frames.data_frames import ImageRawFrame

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
from src.services.mcp_client import MultiMCPClients
from src.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAIAssistantContextAggregator,
    OpenAIUserContextAggregator,
)

from dotenv import load_dotenv

load_dotenv(override=True)


@register_ai_room_bots.register
class DailyMultiMCPBot(DailyRoomBot):
    """
    daily webrtc + asr + llm(gemini) + multi mcp servers + tts bot
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
            camera_out_enabled=False,
            camera_out_width=1024,
            camera_out_height=768,
            # camera_out_is_live=True,
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

        llm_processor: LLMProcessor = self.get_llm_processor()

        mcp_clients = MultiMCPClients(mcp_servers_config=self._bot_config.mcp_servers)
        tools = await mcp_clients.register_tools(llm_processor)
        tools_description = tools.get_tools_description()

        # TODO @weedge:
        # if use json response shot system prompt
        # need support llm assistant response json to exract tool name and arguments
        json_obj_tip = (
            "IMPORTANT: When you need to use a tool, you must ONLY respond with "
            "the exact JSON object format below, nothing else:\n"
            "{\n"
            '    "tool": "tool-name",\n'
            '    "arguments": {\n'
            '        "argument-name": "value"\n'
            "    }\n"
            "}\n"
        )
        system_message = (
            "You are a helpful assistant with access to these tools:\n\n"
            f"{tools_description}\n"
            "Choose the appropriate tool based on the user's question. "
            "If no tool is needed, reply directly.\n\n"
            "After receiving a tool's response:\n"
            "1. Transform the raw data into a natural, conversational response\n"
            "2. Keep responses concise but informative\n"
            "3. Focus on the most relevant information\n"
            "4. Use appropriate context from the user's question\n"
            "5. Avoid simply repeating the raw data\n"
            "6. Your output will be converted to audio so don't include special characters in your answers.\n"
            "7. Respond to what the user said in a creative and helpful way.\n"
            "8. Don't overexplain what you are doing.\n"
            "9. Just respond with short sentences when you are carrying out tool calls.\n\n"
            "Please use only the tools that are explicitly defined above.\n"
        )
        messages = [{"role": "system", "content": system_message}]
        if self._bot_config.llm.messages:
            messages = self._bot_config.llm.messages

        llm_context = OpenAILLMContext()
        llm_context.set_tools(tools)
        llm_context.set_messages(messages)
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
