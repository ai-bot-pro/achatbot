import logging

from fastapi import WebSocket
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner

from src.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator,
)
from src.processors.llm.base import LLMProcessor
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.cmd.bots.bridge.base import AISmallWebRTCFastapiWebsocketBot
from src.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from src.cmd.bots import register_ai_fastapi_ws_bots
from src.types.network.fastapi_websocket import AudioCameraParams, FastapiWebsocketServerParams
from src.transports.fastapi_websocket_server import FastapiWebsocketTransport
from src.services.webrtc_peer_connection import SmallWebRTCConnection
from src.transports.small_webrtc import SmallWebRTCTransport
from src.serializers.avatar_protobuf import AvatarProtobufFrameSerializer
from src.types.frames.data_frames import LLMMessagesFrame


from dotenv import load_dotenv

load_dotenv(override=True)


@register_ai_fastapi_ws_bots.register
class SmallWebRTCFastapiWebsocketAvatarChatBot(AISmallWebRTCFastapiWebsocketBot):
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
        self.asr = self.get_asr()
        assert self.vad_analyzer is not None
        assert self.avatar is not None
        assert self.asr is not None

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

        ws_params = FastapiWebsocketServerParams(
            audio_in_enabled=False,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=self.vad_analyzer,
            vad_audio_passthrough=True,
            transcription_enabled=False,
            add_wav_header=True,
            audio_frame_size=6400,  # output 200ms with 16K hz 1 channel 2 sample_width
        )

        asr_processor = self.get_asr_processor(self.asr)

        llm_processor: LLMProcessor = self.get_llm_processor()
        messages = []
        if self._bot_config.llm.messages:
            messages = self._bot_config.llm.messages
        user_response = LLMUserResponseAggregator(messages)
        assistant_response = LLMAssistantResponseAggregator(messages)

        tts_processor: TTSProcessor = self.get_tts_processor()
        stream_info = tts_processor.get_stream_info()
        ws_params.audio_out_sample_rate = stream_info["sample_rate"]
        ws_params.audio_out_channels = stream_info["channels"]

        avatar_processor = self.get_avatar_processor(self.avatar)

        if (
            self._bot_config.avatar
            and self._bot_config.avatar.tag
            and "audio2expression" in self._bot_config.avatar.tag
        ):
            # audio_expression frame serializer
            ws_params.serializer = AvatarProtobufFrameSerializer()
            self.avatar.args.speaker_audio_sample_rate = ws_params.audio_out_sample_rate
        ws_transport = FastapiWebsocketTransport(
            websocket=self._websocket,
            params=ws_params,
        )

        pipe = Pipeline(
            [
                rtc_transport.input_processor(),
                asr_processor,
                user_response,
                llm_processor,
                tts_processor,
                avatar_processor,
                ws_transport.output_processor(),
                assistant_response,
            ],
        )
        self.task = PipelineTask(
            pipeline=pipe,
            params=PipelineParams(
                allow_interruptions=False,
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

    async def on_ws_client_connected(
        self,
        transport: FastapiWebsocketTransport,
        websocket: WebSocket,
    ):
        logging.info(f"on_ws_client_connected client:{websocket.client}")
        self.session.set_client_id(client_id=f"{websocket.client.host}:{websocket.client.port}")

        # joined use tts say "hello" to introduce with llm generate
        if (
            self._bot_config.tts
            and self._bot_config.llm
            and self._bot_config.llm.messages
            and len(self._bot_config.llm.messages) == 1
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
