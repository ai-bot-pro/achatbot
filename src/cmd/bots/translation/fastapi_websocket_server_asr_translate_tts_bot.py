import logging

from dotenv import load_dotenv
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.parallel_pipeline import ParallelPipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.processors.logger import FrameLogger
from fastapi import WebSocket

from src.cmd.bots.base_fastapi_websocket_server import AIFastapiWebsocketBot
from src.processors.translation.llm_translate_processor import LLMTranslateProcessor
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from src.types.frames.data_frames import TextFrame, TranslationFrame, AudioRawFrame
from src.cmd.bots import register_ai_fastapi_ws_bots
from src.types.network.fastapi_websocket import FastapiWebsocketServerParams
from src.transports.fastapi_websocket_server import FastapiWebsocketTransport
from src.processors.speech.audio_save_processor import SaveAllAudioProcessor
from src.processors.punctuation_processor import PunctuationProcessor
from src.modules.punctuation import PuncEnvInit
from src.serializers.transcription_protobuf import TranscriptionFrameSerializer


load_dotenv(override=True)

"""
TOKENIZERS_PARALLELISM=false python -m src.cmd.websocket.server.fastapi_ws_bot_serve -f config/bots/fastapi_websocket_asr_translate_tts_bot.json
"""


@register_ai_fastapi_ws_bots.register
class FastapiWebsocketServerASRTranslateTTSBot(AIFastapiWebsocketBot):
    """
    fastapi websocket input/output server bot with vad,asr,llm(translate task),punc,tts
    """

    def __init__(self, websocket: WebSocket | None = None, **args) -> None:
        super().__init__(websocket=websocket, **args)
        self.init_bot_config()

        self.vad_analyzer = None
        self.asr_engine = None
        self.generator = None
        self.tts_engine = None
        self.asr_punc_engine = None

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
                self.asr_punc_engine = PuncEnvInit.initEngine(tag, **args)

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
            serializer=TranscriptionFrameSerializer(),
        )
        stream_info = self.tts_engine.get_stream_info()
        self.params.audio_out_sample_rate = stream_info["rate"]
        self.params.audio_out_channels = stream_info["channels"]
        transport = FastapiWebsocketTransport(
            websocket=self._websocket,
            params=self.params,
        )

        asr_processor = self.get_asr_processor(asr_engine=self.asr_engine)

        punc_processor = None
        if self.asr_punc_engine:
            punc_processor = PunctuationProcessor(engine=self.asr_punc_engine, session=self.session)

        tl_processor = None
        if self.generator is not None:
            tl_processor = LLMTranslateProcessor(
                tokenizer=self.get_hf_tokenizer(),
                generator=self.generator,
                session=self.session,
                src=self._bot_config.translate_llm.src,
                target=self._bot_config.translate_llm.target,
                streaming=self._bot_config.translate_llm.streaming,
                prompt_tpl=self._bot_config.translate_llm.prompt_tpl,
            )

        self.tts_processor: TTSProcessor = self.get_tts_processor(tts_engine=self.tts_engine)

        # record_save_processor = SaveAllAudioProcessor(
        #    prefix_name="fastapi_ws_asr_translate_tts_bot",
        #    sample_rate=self.params.audio_in_sample_rate,
        #    channels=self.params.audio_in_channels,
        #    sample_width=self.params.audio_in_sample_width,
        # )
        processors = [
            transport.input_processor(),
            # record_save_processor,
            asr_processor,
            FrameLogger(include_frame_types=[TextFrame]),
            punc_processor,
            FrameLogger(include_frame_types=[TextFrame]),
            ParallelPipeline(
                [transport.output_processor()],
                [
                    tl_processor,
                    FrameLogger(include_frame_types=[TextFrame]),
                    self.tts_processor,
                    FrameLogger(include_frame_types=[TextFrame, AudioRawFrame]),
                    transport.output_processor(),
                ],
            ),
        ]
        processors = [p for p in processors if p is not None]
        logging.info(f"{processors=}")

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

        self.runner = PipelineRunner(handle_sigint=self._handle_sigint)
        await self.runner.run(self.task)

    async def on_client_connected(
        self,
        transport: FastapiWebsocketTransport,
        websocket: WebSocket,
    ):
        logging.info(f"on_client_connected client:{websocket.client}")
        self.session.set_client_id(client_id=f"{websocket.client.host}:{websocket.client.port}")

        if self._bot_config.translate_llm and self._bot_config.translate_llm.init_prompt:
            await self.tts_processor.say(self._bot_config.translate_llm.init_prompt)
