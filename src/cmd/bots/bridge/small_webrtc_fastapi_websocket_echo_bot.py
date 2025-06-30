import logging
from typing import Any

from fastapi import WebSocket
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.frames import EndFrame
from apipeline.pipeline.parallel_pipeline import ParallelPipeline

from src.cmd.bots.bridge.base import AISmallWebRTCFastapiWebsocketBot
from src.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from src.cmd.bots import register_ai_fastapi_ws_bots
from src.types.network.fastapi_websocket import AudioCameraParams, FastapiWebsocketServerParams
from src.transports.fastapi_websocket_server import FastapiWebsocketTransport
from src.services.webrtc_peer_connection import SmallWebRTCConnection
from src.transports.small_webrtc import SmallWebRTCTransport


from dotenv import load_dotenv

load_dotenv(override=True)


@register_ai_fastapi_ws_bots.register
class SmallWebRTCFastapiWebsocketEchoBot(AISmallWebRTCFastapiWebsocketBot):
    """
    don't use MRO (SmallWebrtcAIBot, AIFastapiWebsocketBot)

    - webrtc input (audio) / fastapi websocket output(audio)
    - fastapi websocket input(audio) / webrtc output (audio)
    """

    def __init__(
        self,
        webrtc_connection: SmallWebRTCConnection | None = None,
        websocket: WebSocket | None = None,
        **args,
    ) -> None:
        super().__init__(webrtc_connection, websocket, **args)
        self.vad_analyzer = VADAnalyzerEnvInit.initVADAnalyzerEngine()

    def set_fastapi_websocket(self, websocket: WebSocket):
        self._websocket = websocket

    def set_webrtc_connection(self, webrtc_connection: SmallWebRTCConnection):
        self._webrtc_connection = webrtc_connection

    async def arun(self):
        ws_transport = FastapiWebsocketTransport(
            websocket=self._websocket,
            params=FastapiWebsocketServerParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=True,
                vad_enabled=True,
                vad_analyzer=self.vad_analyzer,
                vad_audio_passthrough=True,
                transcription_enabled=False,
            ),
        )

        rtc_transport = SmallWebRTCTransport(
            webrtc_connection=self._webrtc_connection,
            params=AudioCameraParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_enabled=True,
                vad_analyzer=self.vad_analyzer,
                vad_audio_passthrough=True,
                transcription_enabled=False,
            ),
        )

        pipe = Pipeline(
            [
                ParallelPipeline(
                    [
                        ws_transport.input_processor(),
                        rtc_transport.output_processor(),
                    ],
                    [
                        rtc_transport.input_processor(),
                        ws_transport.output_processor(),
                    ],
                )
            ],
        )
        pipe = Pipeline(
            [
                rtc_transport.input_processor(),
                ws_transport.output_processor(),
            ],
        )
        pipe = Pipeline(
            [
                ws_transport.input_processor(),
                rtc_transport.output_processor(),
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
