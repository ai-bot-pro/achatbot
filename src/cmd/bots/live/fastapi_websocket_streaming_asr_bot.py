import logging
import time
import uuid
import aiofiles

from dotenv import load_dotenv
from fastapi import WebSocket
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.processors.logger import FrameLogger
from apipeline.frames import AudioRawFrame, TextFrame, Frame, CancelFrame, EndFrame
from apipeline.processors.frame_processor import FrameDirection, FrameProcessor

from src.common.session import Session, SessionCtx
from src.serializers.transcription_protobuf import TranscriptionFrameSerializer
from src.processors.speech.vad_audio_save_processor import VADAudioSaveProcessor
from src.modules.speech.asr_live import ASRLiveEnvInit
from src.cmd.bots.base_fastapi_websocket_server import AIFastapiWebsocketBot
from src.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from src.cmd.bots import register_ai_fastapi_ws_bots
from src.types.network.fastapi_websocket import FastapiWebsocketServerParams
from src.transports.fastapi_websocket_server import FastapiWebsocketTransport
from src.processors.speech.asr.asr_live_processor import ASRLiveProcessor
from src.processors.punctuation_processor import PunctuationProcessor
from src.modules.punctuation import PuncEnvInit

load_dotenv(override=True)


class SaveASRText(FrameProcessor):
    def __init__(self, *, is_save=True, name=None, loop=None, **kwargs):
        super().__init__(name=name, loop=loop, **kwargs)
        cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.save_file = f"asr_output_{cur_time}.txt"
        self.asr_text = ""
        self.is_save = is_save

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, TextFrame):
            self.asr_text += frame.text
        if isinstance(frame, (CancelFrame, EndFrame)):
            if not self.is_save:
                return
            # save asr text to file
            async with aiofiles.open(self.save_file, "w", encoding="utf-8") as f:
                await f.write(self.asr_text)


@register_ai_fastapi_ws_bots.register
class FastapiWebsocketStreamingASRBot(AIFastapiWebsocketBot):
    """
    fastapi websocket input(audio)/output(text) server bot with vad,asr,punc,nt
    """

    def __init__(self, websocket: WebSocket | None = None, **args) -> None:
        super().__init__(websocket=websocket, **args)
        self.init_bot_config()

        self.vad_analyzer = None
        self.asr_engine = None
        self.punc_engine = None

    def load(self):
        # load vad analyer
        if self._bot_config.vad:
            tag = self._bot_config.vad.tag
            args = self._bot_config.vad.args or {}
            self.vad_analyzer = VADAnalyzerEnvInit.initVADAnalyzerEngine(tag, args)
        else:
            self.vad_analyzer = VADAnalyzerEnvInit.initVADAnalyzerEngine()

        # load asr live engine
        if self._bot_config.asr:
            tag = self._bot_config.asr.tag
            args = self._bot_config.asr.args or {}
            self.asr_engine = ASRLiveEnvInit.getEngine(tag, **args)
        else:
            logging.info("use defualt asr live engine")
            self.asr_engine = ASRLiveEnvInit.getEngine()

        # load punctuation engine
        if self.asr_engine.get_args_dict().get("textnorm", False) is False:
            if self._bot_config.punctuation:
                tag = self._bot_config.punctuation.tag
                args = self._bot_config.punctuation.args or {}
                self.punc_engine = PuncEnvInit.initEngine(tag, **args)

    async def arun(self):
        if self._websocket is None:
            return

        serializer = TranscriptionFrameSerializer()
        self.params = FastapiWebsocketServerParams(
            audio_in_enabled=True,
            audio_out_enabled=False,
            add_wav_header=True,
            vad_enabled=True,
            vad_analyzer=self.vad_analyzer,
            vad_audio_passthrough=True,
            serializer=serializer,
        )
        transport = FastapiWebsocketTransport(
            websocket=self._websocket,
            params=self.params,
        )

        session = Session(**SessionCtx(str(uuid.uuid4())).__dict__)
        asr_live_processor = ASRLiveProcessor(asr=self.asr_engine, session=session)

        processors = [
            transport.input_processor(),
            # FrameLogger(include_frame_types=[AudioRawFrame]),
            # VADAudioSaveProcessor(prefix_name="streaming_asr_vad", pass_raw_audio=True),
            asr_live_processor,
            FrameLogger(include_frame_types=[TextFrame]),
        ]
        if self.punc_engine:
            punc_processor = PunctuationProcessor(engine=self.punc_engine, session=session)
            processors.append(punc_processor)

        processors.append(FrameLogger(include_frame_types=[TextFrame]))
        processors.append(SaveASRText())
        processors.append(transport.output_processor())

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
