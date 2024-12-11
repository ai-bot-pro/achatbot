import logging

from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from fastapi import WebSocket

from src.cmd.bots.base_fastapi_websocket_server import AIFastapiWebsocketBot
from src.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator,
)
from src.processors.llm.base import LLMProcessor
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from src.types.frames.data_frames import LLMMessagesFrame
from src.cmd.bots import register_ai_fastapi_ws_bots

from dotenv import load_dotenv

from src.types.network.fastapi_websocket import FastapiWebsocketServerParams
from src.transports.fastapi_websocket_server import FastapiWebsocketTransport

load_dotenv(override=True)


@register_ai_fastapi_ws_bots.register
class FastapiWebsocketServerBot(AIFastapiWebsocketBot):
    """
    fastapi websocket input/output server bot with vad,asr,llm,tts
    """

    def __init__(self, websocket: WebSocket | None = None, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()

        self.vad_analyzer = VADAnalyzerEnvInit.initVADAnalyzerEngine()
        self.asr_processor = self.get_asr_processor()
        self.llm_processor: LLMProcessor = self.get_llm_processor()
        self.tts_processor: TTSProcessor = self.get_tts_processor()

    async def arun(self):
        if self._websocket is None:
            return

        self.params = FastapiWebsocketServerParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=True,
            vad_enabled=True,
            vad_analyzer=self.vad_analyzer,
            vad_audio_passthrough=True,
            transcription_enabled=False,
        )
        stream_info = self.tts_processor.get_stream_info()
        self.params.audio_out_sample_rate = stream_info["sample_rate"]
        self.params.audio_out_channels = stream_info["channels"]
        transport = FastapiWebsocketTransport(
            websocket=self._websocket,
            params=self.params,
        )

        messages = []
        if self._bot_config.llm.messages:
            messages = self._bot_config.llm.messages
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
                enable_metrics=True,
                send_initial_empty_metrics=False,
            ),
        )

        transport.add_event_handler("on_client_connected", self.on_client_connected)
        transport.add_event_handler("on_client_disconnected", self.on_client_disconnected)

        await PipelineRunner(handle_sigint=self._handle_sigint).run(self.task)

    async def on_client_connected(
        self,
        transport: FastapiWebsocketTransport,
        websocket: WebSocket,
    ):
        logging.info(f"on_client_connected client:{websocket.client}")
        self.session.set_client_id(client_id=f"{websocket.client.host}:{websocket.client.port}")

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
