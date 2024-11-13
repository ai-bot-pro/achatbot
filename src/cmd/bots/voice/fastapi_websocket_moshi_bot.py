import logging

from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.processors.logger import FrameLogger
from apipeline.frames import AudioRawFrame, TextFrame
from fastapi import WebSocket
from dotenv import load_dotenv

from src.cmd.bots.base import AIRoomBot
from src.processors.voice.moshi_voice_processor import MoshiVoiceOpusStreamProcessor
from src.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from src.cmd.bots import register_ai_room_bots
from src.types.llm.lmgen import LMGenArgs
from src.types.network.fastapi_websocket import FastapiWebsocketServerParams
from src.transports.fastapi_websocket_server import FastapiWebsocketTransport


load_dotenv(override=True)


class FastapiWebsocketMoshiVoiceBot(AIRoomBot):
    """
    use moshi voice processor, just a simple chat bot.
    """

    def __init__(self, websocket: WebSocket | None = None, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()
        self._websocket = websocket

    async def arun(self):
        if self._websocket is None:
            return

        vad_analyzer = VADAnalyzerEnvInit.initVADAnalyzerEngine()
        self.params = FastapiWebsocketServerParams(
            audio_out_enabled=True,
            add_wav_header=True,
            vad_enabled=True,
            vad_analyzer=vad_analyzer,
            vad_audio_passthrough=True,
            transcription_enabled=False,
        )

        voice_processor = MoshiVoiceOpusStreamProcessor(lm_gen_args=LMGenArgs())
        stream_info = voice_processor.stream_info
        self.params.audio_out_sample_rate = stream_info["sample_rate"]
        self.params.audio_out_channels = stream_info["channels"]

        transport = FastapiWebsocketTransport(
            websocket=self._websocket,
            params=self.params,
        )

        messages = []
        if self._bot_config.llm.messages:
            messages = self._bot_config.llm.messages

        self.task = PipelineTask(
            Pipeline([
                transport.input_processor(),
                FrameLogger(include_frame_types=[AudioRawFrame]),
                voice_processor,
                FrameLogger(include_frame_types=[AudioRawFrame, TextFrame]),
                transport.output_processor(),
            ]),
            params=PipelineParams(
                allow_interruptions=False,
                enable_metrics=True,
                send_initial_empty_metrics=False,
            ),
        )

        transport.add_event_handler(
            "on_client_connected",
            self.on_client_connected)
        transport.add_event_handler(
            "on_client_disconnected",
            self.on_client_disconnected)

        await PipelineRunner().run(self.task)

    async def on_client_connected(
        self,
        transport: FastapiWebsocketTransport,
        websocket: WebSocket,
    ):
        logging.info(f"on_client_disconnected client:{websocket.client}")
        self.session.set_client_id(client_id=f"{websocket.client.host}:{websocket.client.port}")

    async def on_client_disconnected(
        self,
        transport: FastapiWebsocketTransport,
        websocket: WebSocket,
    ):
        logging.info(f"on_client_disconnected client:{websocket.client}")
