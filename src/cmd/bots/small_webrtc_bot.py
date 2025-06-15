import logging
from typing import Any

from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner

from src.services.webrtc_peer_connection import SmallWebRTCConnection
from src.cmd.bots.base_small_webrtc import SmallWebrtcAIBot
from src.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator,
)
from src.processors.llm.base import LLMProcessor
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from src.types.frames.data_frames import LLMMessagesFrame, TransportMessageFrame
from src.cmd.bots import register_ai_small_webrtc_bots
from src.common.types import AudioCameraParams
from src.transports.small_webrtc import SmallWebRTCTransport

from dotenv import load_dotenv

load_dotenv(override=True)


@register_ai_small_webrtc_bots.register
class SmallWebrtcBot(SmallWebrtcAIBot):
    """
    webrtc input/output server bot with vad,asr,llm,tts
    """

    def __init__(self, webrtc_connection: SmallWebRTCConnection | None = None, **args) -> None:
        super().__init__(webrtc_connection=webrtc_connection, **args)
        self.init_bot_config()

        self.vad_analyzer = VADAnalyzerEnvInit.initVADAnalyzerEngine()
        self.asr_processor = self.get_asr_processor()
        self.llm_processor: LLMProcessor = self.get_llm_processor()
        self.tts_processor: TTSProcessor = self.get_tts_processor()

    async def arun(self):
        if self._webrtc_connection is None:
            return

        self.params = AudioCameraParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=self.vad_analyzer,
            vad_audio_passthrough=True,
        )
        stream_info = self.tts_processor.get_stream_info()
        self.params.audio_out_sample_rate = stream_info["sample_rate"]
        self.params.audio_out_channels = stream_info["channels"]
        transport = SmallWebRTCTransport(
            webrtc_connection=self._webrtc_connection,
            params=self.params,
        )
        self.register_event(transport)

        messages = (
            list(self._bot_config.llm.messages)
            if self._bot_config.llm and self._bot_config.llm.messages
            else []
        )
        user_response = LLMUserResponseAggregator(messages)
        assistant_response = LLMAssistantResponseAggregator(messages)

        self.task = PipelineTask(
            Pipeline(
                [
                    transport.input_processor(),
                    self.asr_processor,
                    user_response,
                    self.llm_processor,
                    self.tts_processor,
                    transport.output_processor(),
                    assistant_response,
                ]
            ),
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=False,
                send_initial_empty_metrics=False,
            ),
        )

        # NOTE: if bot run in the sub thread like fastapi/starlette background-tasks, handle_sigint set False
        await PipelineRunner(handle_sigint=False).run(self.task)

    async def on_client_connected(
        self,
        transport: SmallWebRTCTransport,
        connection: SmallWebRTCConnection,
    ):
        logging.info(f"on_client_connected {connection.pc_id=} {connection.connectionState=}")
        self.session.set_client_id(connection.pc_id)
        message = TransportMessageFrame(
            message={"type": "meta", "protocol": "small-webrtc", "version": "0.0.1"},
            urgent=True,
        )
        await transport.output_processor().send_message(message)

        # joined use tts say "hello" to introduce with llm generate
        if self._bot_config.tts and self._bot_config.llm and self._bot_config.llm.messages:
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

    async def on_app_message(
        self,
        transport: SmallWebRTCTransport,
        connection: SmallWebRTCConnection,
        message: Any,
    ):
        logging.info(f"on_app_message received message: {message}")