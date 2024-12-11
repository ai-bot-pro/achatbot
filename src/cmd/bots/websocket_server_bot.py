import logging

from apipeline.frames import EndFrame
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
import websockets
from dotenv import load_dotenv

from src.cmd.bots.base import AIRoomBot
from src.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator,
)
from src.processors.llm.base import LLMProcessor
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from src.types.frames.data_frames import LLMMessagesFrame
from src.cmd.bots import register_ai_room_bots
from src.transports.websocket_server import WebsocketServerTransport
from src.types.network.websocket import WebsocketServerParams

load_dotenv(override=True)


@register_ai_room_bots.register
class WebsocketServerBot(AIRoomBot):
    """
    websocket input/output server bot with vad,asr,llm,tts
    """

    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()

    async def arun(self):
        vad_analyzer = VADAnalyzerEnvInit.initVADAnalyzerEngine()
        self.params = WebsocketServerParams(
            audio_out_enabled=True,
            add_wav_header=True,
            vad_enabled=True,
            vad_analyzer=vad_analyzer,
            vad_audio_passthrough=True,
            transcription_enabled=False,
        )
        asr_processor = self.get_asr_processor()
        llm_processor: LLMProcessor = self.get_llm_processor()
        tts_processor: TTSProcessor = self.get_tts_processor()
        stream_info = tts_processor.get_stream_info()
        self.params.audio_out_sample_rate = stream_info["sample_rate"]
        self.params.audio_out_channels = stream_info["channels"]
        transport = WebsocketServerTransport(
            host=self.args.websocket_server_host,
            port=self.args.websocket_server_port,
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
                    asr_processor,
                    user_response,
                    llm_processor,
                    tts_processor,
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

        await PipelineRunner().run(self.task)

    async def on_client_connected(
        self, transport: WebsocketServerTransport, client: websockets.WebSocketServerProtocol
    ):
        logging.info(f"on_client_disconnected client:{client}")
        self.session.set_client_id(client_id=client.id)

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

    async def on_client_disconnected(
        self, transport: WebsocketServerTransport, client: websockets.WebSocketServerProtocol
    ):
        logging.info(f"on_client_disconnected client:{client}")
        if self.task is not None:
            await self.task.queue_frame(EndFrame())
