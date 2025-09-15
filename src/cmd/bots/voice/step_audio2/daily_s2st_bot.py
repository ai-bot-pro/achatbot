import logging
import copy

from dotenv import load_dotenv
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.parallel_pipeline import ParallelPipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.processors.logger import FrameLogger

from src.processors.speech.audio_save_processor import AudioSaveProcessor, SaveAllAudioProcessor
from src.processors.aggregators.user_audio_response import UserAudioResponseAggregator
from src.types.frames.data_frames import (
    TextFrame,
    AudioRawFrame,
    PathAudioRawFrame,
    LLMGenedTokensFrame,
)
from src.common.types import DailyParams
from src.transports.daily import DailyTransport
from src.cmd.bots.base_daily import DailyRoomBot
from src.cmd.bots import register_ai_room_bots
from src.processors.punctuation_processor import PunctuationProcessor
from src.modules.punctuation import PuncEnvInit
from .helper import get_step_audio2_processor, get_step_audio2_llm


load_dotenv(override=True)

"""
TOKENIZERS_PARALLELISM=false python -m src.cmd.bots.main -f config/bots/daily_step_audio2_s2st_bot.json
"""


@register_ai_room_bots.register
class DailyStepAudio2S2STBot(DailyRoomBot):
    """
    daily input/output server bot with step2_audio(translate task)
    """

    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()

        self.vad_analyzer = None
        self.audio_llm = None
        self.asr_punc_engine = None

    def load(self):
        self.vad_analyzer = self.get_vad_analyzer()
        llm_conf = self._bot_config.voice_llm or self._bot_config.asr
        if llm_conf:
            self.audio_llm = get_step_audio2_llm(llm_conf)

        # load punctuation engine
        if self._bot_config.punctuation:
            tag = self._bot_config.punctuation.tag
            args = self._bot_config.punctuation.args or {}
            self.asr_punc_engine = PuncEnvInit.initEngine(tag, **args)

    async def arun(self):
        assert self.vad_analyzer is not None

        self.params = DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=self.vad_analyzer,
            vad_audio_passthrough=True,
        )

        self._voice_processor = None
        if self._bot_config.voice_llm and self.audio_llm:
            # src/processors/voice/step_audio2_processor.py
            self._voice_processor = get_step_audio2_processor(
                self._bot_config.voice_llm,
                session=copy.deepcopy(self.session),
                audio_llm=self.audio_llm,
                processor_class_name="StepS2STProcessor",
            )
            if hasattr(self._voice_processor, "stream_info"):
                stream_info = self._voice_processor.stream_info
                self.params.audio_out_sample_rate = stream_info["sample_rate"]
                self.params.audio_out_channels = stream_info["channels"]
        logging.info(f"params: {self.params}")
        transport = DailyTransport(
            self.args.room_url,
            self.args.token,
            self.args.bot_name,
            self.params,
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
                prefix_name="daily_s2st_bot",
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

        transport.add_event_handlers(
            "on_first_participant_joined",
            [self.on_first_participant_joined, self.on_first_participant_say_hi],
        )
        transport.add_event_handler("on_participant_left", self.on_participant_left)
        transport.add_event_handler("on_call_state_updated", self.on_call_state_updated)

        self.runner = PipelineRunner(handle_sigint=self._handle_sigint)
        await self.runner.run(self.task)

    async def on_first_participant_say_hi(
        self,
        transport: DailyTransport,
        participant,
    ):
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
