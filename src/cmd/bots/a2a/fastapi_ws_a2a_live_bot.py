import logging
from typing import cast

from fastapi import WebSocket
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.frames import TextFrame

from src.cmd.bots.base_fastapi_websocket_server import AIFastapiWebsocketBot
from src.types.network.fastapi_websocket import FastapiWebsocketServerParams
from src.transports.fastapi_websocket_server import FastapiWebsocketTransport
from src.serializers.transcription_protobuf import TranscriptionFrameSerializer
from src.cmd.bots import register_ai_fastapi_ws_bots
from src.processors.a2a.a2a_live_processor import A2ALiveProcessor

from dotenv import load_dotenv

load_dotenv(override=True)


@register_ai_fastapi_ws_bots.register
class FastapiWebsocketA2ALiveBot(AIFastapiWebsocketBot):
    """
    use a2a live bot
    - no VAD
    - text/images/audio -> gemini live (BIDI) -> text/audio
    """

    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()

    async def arun(self):
        if self._websocket is None:
            return

        serializer = TranscriptionFrameSerializer()
        self.params = FastapiWebsocketServerParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=True,
            vad_enabled=False,
            vad_audio_passthrough=True,
            serializer=serializer,
            audio_out_sample_rate=24000,
        )
        transport = FastapiWebsocketTransport(
            websocket=self._websocket,
            params=self.params,
        )
        self.a2a_processor: A2ALiveProcessor = cast(
            A2ALiveProcessor,
            self.get_a2a_processor(tag="a2a_live_processor"),
        )

        self.task = PipelineTask(
            Pipeline(
                [
                    transport.input_processor(),
                    self.a2a_processor,
                    transport.output_processor(),
                ]
            ),
            params=PipelineParams(
                allow_interruptions=False,
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
        client_id = f"{websocket.client.host}:{websocket.client.port}"
        self.session.set_client_id(client_id=client_id)
        self.a2a_processor.set_user_id(client_id)
        await self.a2a_processor.create_conversation()
        await self.a2a_processor.create_push_task()

        is_cn = self._bot_config.a2a and self._bot_config.a2a.language == "zh"
        user_hi_text = "请用中文介绍下自己。" if is_cn else "Please introduce yourself first."
        await self.task.queue_frame(TextFrame(text=user_hi_text))
