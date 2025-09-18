import logging
import copy

from dotenv import load_dotenv
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.parallel_pipeline import ParallelPipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.processors.logger import FrameLogger
from fastapi import WebSocket

from src.cmd.bots.base_fastapi_websocket_server import AIFastapiWebsocketBot
from src.processors.speech.audio_save_processor import AudioSaveProcessor, SaveAllAudioProcessor
from src.processors.aggregators.user_audio_response import UserAudioResponseAggregator
from src.types.frames.data_frames import (
    TextFrame,
    AudioRawFrame,
    PathAudioRawFrame,
    LLMGenedTokensFrame,
)
from src.cmd.bots import register_ai_fastapi_ws_bots
from src.types.network.fastapi_websocket import FastapiWebsocketServerParams
from src.transports.fastapi_websocket_server import FastapiWebsocketTransport
from src.processors.punctuation_processor import PunctuationProcessor
from src.modules.punctuation import PuncEnvInit
from src.serializers.transcription_protobuf import TranscriptionFrameSerializer
from .helper import get_step_audio2_processor, get_step_audio2_llm, get_token2wav


load_dotenv(override=True)

"""
TOKENIZERS_PARALLELISM=false python -m src.cmd.websocket.server.fastapi_ws_bot_serve -f config/bots/fastapi_websocket_step_audio2_s2st_bot.json
"""


@register_ai_fastapi_ws_bots.register
class FastapiWebsocketServerStepAudio2S2STBot(AIFastapiWebsocketBot):
    """
    fastapi websocket input/output server bot with step2_audio(translate task)
    """

    def __init__(self, websocket: WebSocket | None = None, **args) -> None:
        super().__init__(websocket=websocket, **args)
        self.init_bot_config()

        self.vad_analyzer = None
        self.asr_punc_engine = None
        self.audio_llm = None
        self.token2wav = None

    def load(self):
        self.vad_analyzer = self.get_vad_analyzer()

        # load punctuation engine
        if self._bot_config.punctuation:
            tag = self._bot_config.punctuation.tag
            args = self._bot_config.punctuation.args or {}
            self.asr_punc_engine = PuncEnvInit.initEngine(tag, **args)

        llm_conf = self._bot_config.voice_llm or self._bot_config.asr
        if llm_conf:
            self.audio_llm = get_step_audio2_llm(llm_conf)
            # self.token2wav = get_token2wav(llm_conf)

    async def arun(self):
        if self._websocket is None:
            return
        assert self.vad_analyzer is not None

        self.params = FastapiWebsocketServerParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=True,
            vad_enabled=True,
            vad_analyzer=self.vad_analyzer,
            vad_audio_passthrough=True,
            serializer=TranscriptionFrameSerializer(),
        )

        self._voice_processor = None
        if self._bot_config.voice_llm and self.audio_llm:
            # src/processors/voice/step_audio2_processor.py
            self._voice_processor = get_step_audio2_processor(
                self._bot_config.voice_llm,
                session=copy.deepcopy(self.session),
                token2wav=self.token2wav,
                audio_llm=self.audio_llm,
                processor_class_name="StepS2STProcessor",
            )
            if hasattr(self._voice_processor, "stream_info"):
                stream_info = self._voice_processor.stream_info
                self.params.audio_out_sample_rate = stream_info["sample_rate"]
                self.params.audio_out_channels = stream_info["channels"]
        logging.info(f"params: {self.params}")
        transport = FastapiWebsocketTransport(
            websocket=self._websocket,
            params=self.params,
        )

        asr_processors = []
        if self._bot_config.asr:
            # src/processors/voice/step_audio2_processor.py
            asr_processor = get_step_audio2_processor(
                self._bot_config.asr,
                session=copy.deepcopy(self.session),
                audio_llm=self.audio_llm,
                processor_class_name="StepASRProcessor",
            )

            punc_processor = None
            if self.asr_punc_engine:
                punc_processor = PunctuationProcessor(
                    engine=self.asr_punc_engine, session=self.session
                )

            asr_processors = [
                asr_processor,
                punc_processor,
                FrameLogger(include_frame_types=[TextFrame]),
                transport.output_processor(),
            ]
            asr_processors = [p for p in asr_processors if p is not None]
            logging.info(f"{asr_processors=}")

        stst_processors = []
        if self._voice_processor:
            save_bot_audio_processor = (
                AudioSaveProcessor(prefix_name="bot_speak", pass_raw_audio=True)
                if self._save_audio
                else None
            )
            stst_processors = [
                self._voice_processor,
                FrameLogger(include_frame_types=[TextFrame, AudioRawFrame, LLMGenedTokensFrame]),
                save_bot_audio_processor,
                transport.output_processor(),
            ]
            stst_processors = [p for p in stst_processors if p is not None]
            logging.info(f"{stst_processors=}")

        save_all_records_processor = (
            SaveAllAudioProcessor(
                prefix_name="fastapi_ws_s2st_bot",
                sample_rate=self.params.audio_in_sample_rate,
                channels=self.params.audio_in_channels,
                sample_width=self.params.audio_in_sample_width,
            )
            if self._save_audio
            else None
        )
        save_speaker_audio_processor = (
            AudioSaveProcessor(prefix_name="user_speak", pass_raw_audio=True)
            if self._save_audio
            else None
        )
        processors = [
            transport.input_processor(),
            save_all_records_processor,
            UserAudioResponseAggregator(),
            FrameLogger(include_frame_types=[AudioRawFrame]),
            save_speaker_audio_processor,
            FrameLogger(include_frame_types=[AudioRawFrame]),
            ParallelPipeline(
                asr_processors,
                stst_processors,
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

        llm_conf = self._bot_config.voice_llm or self._bot_config.llm
        if llm_conf and llm_conf.init_prompt and self._voice_processor:
            await self._voice_processor.say(
                llm_conf.init_prompt,
                temperature=0.1,
                max_new_tokens=1024,
                top_k=20,
                top_p=0.95,
                repetition_penalty=1.1,
            )
