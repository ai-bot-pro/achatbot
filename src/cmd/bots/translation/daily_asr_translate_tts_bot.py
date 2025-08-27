import logging

import uuid
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.processors.logger import FrameLogger

from src.processors.speech.tts.tts_processor import TTSProcessor
from src.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from src.modules.speech.asr_live import ASRLiveEnvInit
from src.common.types import DailyParams
from src.transports.daily import DailyTransport
from src.cmd.bots.base_daily import DailyRoomBot
from src.cmd.bots import register_ai_room_bots
from src.types.frames.data_frames import TextFrame, AudioRawFrame
from src.core.llm import LLMEnvInit
from src.processors.translation.llm_translate_processor import LLMTranslateProcessor
from src.common.session import Session, SessionCtx
from src.processors.punctuation_processor import PunctuationProcessor
from src.modules.punctuation import PuncEnvInit

from dotenv import load_dotenv

load_dotenv(override=True)


@register_ai_room_bots.register
class DailyASRTranslateTTSBot(DailyRoomBot):
    """
    daily transport(webrtc) with vad -> asr -> translate LLM -> punc -> tts
    """

    def __init__(self, **args) -> None:
        super().__init__(**args)
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
        self.tokenizer, self.generator = self.get_translate_llm_generator()

        # load punctuation engine
        if self.asr_engine.get_args_dict().get("textnorm", False) is False:
            if self._bot_config.punctuation:
                tag = self._bot_config.punctuation.tag
                args = self._bot_config.punctuation.args or {}
                self.punc_engine = PuncEnvInit.initEngine(tag, **args)

    async def arun(self):
        self.daily_params = DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=self.vad_analyzer,
            vad_audio_passthrough=True,
            transcription_enabled=False,
        )

        processors = []
        asr_processor = self.get_asr_processor(asr_engine=self.asr_engine)
        processors.append(asr_processor)
        processors.append(FrameLogger(include_frame_types=[TextFrame]))

        if self.tokenizer is not None and self.generator is not None:
            tl_processor = LLMTranslateProcessor(
                tokenizer=self.tokenizer,
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

        stream_info = self.tts_processor.get_stream_info()
        self.daily_params.audio_out_sample_rate = stream_info["sample_rate"]
        self.daily_params.audio_out_channels = stream_info["channels"]

        transport = DailyTransport(
            self.args.room_url,
            self.args.token,
            self.args.bot_name,
            self.daily_params,
        )

        processors = (
            [
                transport.input_processor(),
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

        transport.add_event_handlers(
            "on_first_participant_joined",
            [self.on_first_participant_joined, self.on_first_participant_say_hi],
        )
        transport.add_event_handler("on_participant_left", self.on_participant_left)
        transport.add_event_handler("on_call_state_updated", self.on_call_state_updated)

        await PipelineRunner(handle_sigint=self.args.handle_sigint).run(self.task)

    async def on_first_participant_say_hi(self, transport: DailyTransport, participant):
        self.session.set_client_id(participant["id"])
        if self.daily_params.transcription_enabled:
            transport.capture_participant_transcription(participant["id"])
        await self.tts_processor.say("hi, welcome to chat with translation bot.")
