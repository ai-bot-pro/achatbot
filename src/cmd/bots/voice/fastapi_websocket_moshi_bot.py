import logging

from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.processors.logger import FrameLogger
from apipeline.frames import AudioRawFrame, TextFrame
from fastapi import WebSocket
from dotenv import load_dotenv

from src.cmd.bots.base_fastapi_websocket_server import AIFastapiWebsocketBot
from src.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from src.cmd.bots import register_ai_fastapi_ws_bots
from src.types.network.fastapi_websocket import FastapiWebsocketServerParams
from src.transports.fastapi_websocket_server import FastapiWebsocketTransport


load_dotenv(override=True)


@register_ai_fastapi_ws_bots.register
class FastapiWebsocketMoshiVoiceBot(AIFastapiWebsocketBot):
    """
    use moshi voice processor, just a simple chat bot.
    """

    def __init__(self, websocket: WebSocket | None = None, **args) -> None:
        super().__init__(websocket, **args)
        self.init_bot_config()
        self._voice_processor = self.get_moshi_voice_processor()
        self._vad_analyzer = VADAnalyzerEnvInit.initVADAnalyzerEngine()

    async def arun(self):
        if self._websocket is None:
            return

        self.params = FastapiWebsocketServerParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,  # no wav header, if use opus codec format
            vad_enabled=False,
            vad_analyzer=self._vad_analyzer,
            vad_audio_passthrough=True,
            transcription_enabled=False,
        )

        stream_info = self._voice_processor.stream_info
        self.params.audio_out_sample_rate = stream_info["sample_rate"]
        self.params.audio_out_channels = stream_info["channels"]

        transport = FastapiWebsocketTransport(
            websocket=self._websocket,
            params=self.params,
        )

        # messages = []
        # if self._bot_config.llm.messages:
        #    messages = self._bot_config.llm.messages

        self.task = PipelineTask(
            Pipeline(
                [
                    transport.input_processor(),
                    # FrameLogger(include_frame_types=[AudioRawFrame]),
                    self._voice_processor,
                    FrameLogger(include_frame_types=[AudioRawFrame, TextFrame]),
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
