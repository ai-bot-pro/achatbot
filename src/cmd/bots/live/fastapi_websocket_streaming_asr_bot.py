import logging

from dotenv import load_dotenv
from fastapi import WebSocket
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.processors.logger import FrameLogger
from apipeline.frames import AudioRawFrame, TextFrame

from src.processors.speech.audio_save_processor import AudioSaveProcessor
from src.modules.speech.asr_live import ASRLiveEnvInit
from src.cmd.bots.base_fastapi_websocket_server import AIFastapiWebsocketBot
from src.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from src.cmd.bots import register_ai_fastapi_ws_bots
from src.types.network.fastapi_websocket import FastapiWebsocketServerParams
from src.transports.fastapi_websocket_server import FastapiWebsocketTransport
from src.processors.speech.asr.asr_live_processor import ASRLiveProcessor

load_dotenv(override=True)


@register_ai_fastapi_ws_bots.register
class FastapiWebsocketStreamingASRBot(AIFastapiWebsocketBot):
    """
    fastapi websocket input(audio)/output(text) server bot with vad,asr,punc,nt
    """

    def __init__(self, websocket: WebSocket | None = None, **args) -> None:
        super().__init__(websocket=websocket, **args)
        self.init_bot_config()

        self.vad_analyzer = None
        self.engine = None

    def load(self):
        # load vad analyer
        self.vad_analyzer = VADAnalyzerEnvInit.initVADAnalyzerEngine()

        # load asr live engine
        if self._bot_config.asr and self._bot_config.asr.tag and self._bot_config.asr.args:
            self.engine = ASRLiveEnvInit.getEngine(
                self._bot_config.asr.tag, **self._bot_config.asr.args
            )
        else:
            logging.info("use default asr live engine")
            self.engine = ASRLiveEnvInit.initEngine()

    async def arun(self):
        if self._websocket is None:
            return

        self.params = FastapiWebsocketServerParams(
            audio_in_enabled=True,
            audio_out_enabled=False,
            add_wav_header=True,
            vad_enabled=True,
            vad_analyzer=self.vad_analyzer,
            vad_audio_passthrough=True,
        )
        transport = FastapiWebsocketTransport(
            websocket=self._websocket,
            params=self.params,
        )

        asr_live_processor = ASRLiveProcessor(asr=self.engine)

        self.task = PipelineTask(
            Pipeline(
                [
                    transport.input_processor(),
                    # AudioSaveProcessor(prefix_name="streaming_asr", pass_raw_audio=True),
                    # FrameLogger(include_frame_types=[AudioRawFrame]),
                    asr_live_processor,
                    FrameLogger(include_frame_types=[TextFrame]),
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
