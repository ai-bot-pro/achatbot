import logging
from typing import Any

from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.processors.logger import FrameLogger

from src.services.webrtc_peer_connection import SmallWebRTCConnection
from src.cmd.bots.base_small_webrtc import SmallWebrtcAIBot
from src.processors.translation.llm_translate_processor import LLMTranslateProcessor
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from src.types.frames.data_frames import TextFrame, AudioRawFrame, TransportMessageFrame
from src.cmd.bots import register_ai_small_webrtc_bots
from src.common.types import AudioCameraParams
from src.transports.small_webrtc import SmallWebRTCTransport
from src.processors.punctuation_processor import PunctuationProcessor
from src.modules.punctuation import PuncEnvInit

from dotenv import load_dotenv

load_dotenv(override=True)


@register_ai_small_webrtc_bots.register
class SmallWebrtcASRTranslateTTSBot(SmallWebrtcAIBot):
    """
    webrtc input/output server bot with vad,asr,llm(translate),punc,tts
    """

    def __init__(self, webrtc_connection: SmallWebRTCConnection | None = None, **args) -> None:
        super().__init__(webrtc_connection=webrtc_connection, **args)
        self.init_bot_config()

        self.vad_analyzer = None
        self.asr_engine = None
        self.generator = None
        self.tokenizer = None  # text tokenizer
        self.tts_engine = None
        self.punc_engine = None

    def load(self):
        self.vad_analyzer = self.get_vad_analyzer()
        self.asr_engine = self.get_asr()
        self.tts_engine = self.get_tts()
        self.generator = self.get_translate_llm_generator()

        # load punctuation engine
        if self.asr_engine.get_args_dict().get("textnorm", False) is False:
            if self._bot_config.punctuation:
                tag = self._bot_config.punctuation.tag
                args = self._bot_config.punctuation.args or {}
                self.punc_engine = PuncEnvInit.initEngine(tag, **args)

    async def arun(self):
        if self._webrtc_connection is None:
            return

        processors = []
        asr_processor = self.get_asr_processor(asr_engine=self.asr_engine)
        processors.append(asr_processor)
        processors.append(FrameLogger(include_frame_types=[TextFrame]))

        if self.generator is not None:
            tl_processor = LLMTranslateProcessor(
                tokenizer=self.get_hf_tokenizer(),
                generator=self.generator,
                session=self.session,
                src=self._bot_config.translate_llm.src,
                target=self._bot_config.translate_llm.target,
                streaming=self._bot_config.translate_llm.streaming,
            )
            processors.append(tl_processor)

        if self.punc_engine:
            punc_processor = PunctuationProcessor(engine=self.punc_engine, session=self.session)
            processors.append(punc_processor)
            processors.append(FrameLogger(include_frame_types=[TextFrame]))

        self.tts_processor: TTSProcessor = self.get_tts_processor(tts_engine=self.tts_engine)
        processors.append(self.tts_processor)

        self.params = AudioCameraParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=self.vad_analyzer,
            vad_audio_passthrough=True,
        )
        stream_info = self.tts_processor.get_stream_info()
        self.params.audio_out_sample_rate = stream_info["sample_rate"]
        self.params.audio_out_channels = stream_info["channels"]
        transport = SmallWebRTCTransport(
            webrtc_connection=self._webrtc_connection,
            params=self.params,
        )
        self.register_event(transport)

        processors = (
            [
                transport.input_processor(),
                FrameLogger(include_frame_types=[TextFrame, AudioRawFrame]),
            ]
            + processors
            + [
                FrameLogger(include_frame_types=[TextFrame, AudioRawFrame]),
                transport.output_processor(),
            ]
        )
        logging.info(f"processors: {processors}")

        self.task = PipelineTask(
            Pipeline(processors=processors),
            params=PipelineParams(
                allow_interruptions=False,
                enable_metrics=True,
                send_initial_empty_metrics=False,
            ),
        )

        # NOTE: if bot run in the sub thread like fastapi/starlette background-tasks, handle_sigint set False
        await PipelineRunner(handle_sigint=self._handle_sigint).run(self.task)

    async def on_client_connected(
        self,
        transport: SmallWebRTCTransport,
        connection: SmallWebRTCConnection,
    ):
        logging.info(f"on_client_connected {connection.pc_id=} {connection.connectionState=}")
        self.session.set_client_id(connection.pc_id)
        message = TransportMessageFrame(
            message={"type": "meta", "protocol": "small-webrtc", "version": "0.0.1"},
            urgent=True,
        )
        await transport.output_processor().send_message(message)

        await self.tts_processor.say("hi, welcome to chat with translation bot.")

    async def on_app_message(
        self,
        transport: SmallWebRTCTransport,
        connection: SmallWebRTCConnection,
        message: Any,
    ):
        logging.info(f"on_app_message received message: {message}")
