import logging

from dotenv import load_dotenv
from fastapi import WebSocket
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.processors.logger import FrameLogger
from apipeline.frames.data_frames import AudioRawFrame

from src.processors.speech.audio_save_processor import AudioSaveProcessor
from src.processors.aggregators.user_audio_response import UserAudioResponseAggregator
from src.common.session import Session, SessionCtx
from src.cmd.bots.base_fastapi_websocket_server import AIFastapiWebsocketBot
from src.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from src.cmd.bots import register_ai_fastapi_ws_bots
from src.types.network.fastapi_websocket import FastapiWebsocketServerParams
from src.transports.fastapi_websocket_server import FastapiWebsocketTransport
from src.types.frames import BotSpeakingFrame

load_dotenv(override=True)

"""
TEN_VAD_LIB_PATH=<path_to_ten_vad_library> python -m src.cmd.websocket.server.fastapi_ws_bot_serve -f config/bots/fastapi_websocket_ten_vad_bot.json
"""


@register_ai_fastapi_ws_bots.register
class FastapiWebsocketVADRBot(AIFastapiWebsocketBot):
    """
    fastapi websocket input(audio)/output(text) server bot with vad
    """

    def __init__(self, websocket: WebSocket | None = None, **args) -> None:
        super().__init__(websocket=websocket, **args)
        self.init_bot_config()

        self.vad_analyzer = None

    def load(self):
        # load vad analyer
        if self._bot_config.vad:
            tag = self._bot_config.vad.tag
            args = self._bot_config.vad.args or {}
            self.vad_analyzer = VADAnalyzerEnvInit.initVADAnalyzerEngine(tag, args)
        else:
            self.vad_analyzer = VADAnalyzerEnvInit.initVADAnalyzerEngine()
        logging.info(f"{__class__.__name__} load vad_analyzer: {self.vad_analyzer} ok")

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
        )
        transport = FastapiWebsocketTransport(
            websocket=self._websocket,
            params=self.params,
        )

        in_audio_aggr = UserAudioResponseAggregator()
        save_user_audio_processor = None
        if self._save_audio:
            save_user_audio_processor = AudioSaveProcessor(prefix_name="user", pass_raw_audio=True)

        processors = [
            FrameLogger(include_frame_types=[BotSpeakingFrame]),
            transport.input_processor(),
            in_audio_aggr,
            FrameLogger(include_frame_types=[AudioRawFrame]),
            save_user_audio_processor,
            transport.output_processor(),
        ]

        self.task = PipelineTask(
            Pipeline(processors=processors),
            params=PipelineParams(
                allow_interruptions=False,
                enable_metrics=True,
                send_initial_empty_metrics=False,
            ),
        )

        transport.add_event_handler("on_client_connected", self.on_client_connected)
        transport.add_event_handler("on_client_disconnected", self.on_client_disconnected)

        await PipelineRunner(handle_sigint=self._handle_sigint).run(self.task)
