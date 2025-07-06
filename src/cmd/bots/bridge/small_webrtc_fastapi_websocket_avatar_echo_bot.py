import asyncio

from fastapi import WebSocket
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner

from src.cmd.bots.bridge.base import AISmallWebRTCFastapiWebsocketBot
from src.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from src.cmd.bots import register_ai_fastapi_ws_bots
from src.types.network.fastapi_websocket import AudioCameraParams, FastapiWebsocketServerParams
from src.transports.fastapi_websocket_server import FastapiWebsocketTransport
from src.services.webrtc_peer_connection import SmallWebRTCConnection
from src.transports.small_webrtc import SmallWebRTCTransport
from src.serializers.avatar_protobuf import AvatarProtobufFrameSerializer


from dotenv import load_dotenv

load_dotenv(override=True)


@register_ai_fastapi_ws_bots.register
class SmallWebRTCFastapiWebsocketAvatarEchoBot(AISmallWebRTCFastapiWebsocketBot):
    """
    - webrtc input (vision/audio) / fastapi websocket output(vision/audio/audio_expression)
    """

    def __init__(
        self,
        webrtc_connection: SmallWebRTCConnection | None = None,
        websocket: WebSocket | None = None,
        **args,
    ) -> None:
        super().__init__(webrtc_connection, websocket, **args)
        self.vad_analyzer = None
        self.avatar = None

    def set_fastapi_websocket(self, websocket: WebSocket):
        self._websocket = websocket

    def set_webrtc_connection(self, webrtc_connection: SmallWebRTCConnection):
        self._webrtc_connection = webrtc_connection

    def load(self):
        pass

    async def arun(self):
        self.vad_analyzer = VADAnalyzerEnvInit.initVADAnalyzerEngine()
        self.avatar = self.get_avatar()

        assert self.vad_analyzer is not None
        assert self.avatar is not None

        rtc_transport = SmallWebRTCTransport(
            webrtc_connection=self._webrtc_connection,
            params=AudioCameraParams(
                audio_in_enabled=True,
                audio_out_enabled=False,
                vad_enabled=True,
                vad_analyzer=self.vad_analyzer,
                vad_audio_passthrough=True,
                transcription_enabled=False,
                camera_in_enabled=True,
            ),
        )

        params = FastapiWebsocketServerParams(
            audio_in_enabled=False,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=self.vad_analyzer,
            vad_audio_passthrough=True,
            transcription_enabled=False,
            add_wav_header=True,
            audio_frame_size=6400,  # output 200ms with 16K hz 1 channel 2 sample_width
        )
        if (
            self._bot_config.avatar
            and self._bot_config.avatar.tag
            and "audio2expression" in self._bot_config.avatar.tag
        ):
            # audio_expression frame serializer
            params.serializer = AvatarProtobufFrameSerializer()
        ws_transport = FastapiWebsocketTransport(
            websocket=self._websocket,
            params=params,
            loop=asyncio.get_running_loop(),
        )

        avatar_processor = self.get_avatar_processor(self.avatar)

        pipe = Pipeline(
            [
                rtc_transport.input_processor(),
                avatar_processor,
                ws_transport.output_processor(),
            ],
        )
        self.task = PipelineTask(
            pipeline=pipe,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                send_initial_empty_metrics=False,
            ),
        )

        ws_transport.add_event_handler("on_client_connected", self.on_ws_client_connected)
        ws_transport.add_event_handler("on_client_disconnected", self.on_ws_client_disconnected)
        rtc_transport.add_event_handler("on_client_connected", self.on_rtc_client_connected)
        rtc_transport.add_event_handler("on_client_disconnected", self.on_rtc_client_disconnected)
        rtc_transport.add_event_handler("on_app_message", self.on_rtc_app_message)

        await PipelineRunner(handle_sigint=self._handle_sigint).run(self.task)
