import logging

from dotenv import load_dotenv
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.processors.logger import FrameLogger
from apipeline.frames import AudioRawFrame, TextFrame

from src.processors.speech.audio_save_processor import AudioSaveProcessor
from src.processors.filters.audio_noise_filter import AudioNoiseFilter
from src.processors.aggregators.user_audio_response import UserAudioResponseAggregator
from src.cmd.bots.base_daily import DailyRoomBot
from src.common.types import DailyParams
from src.transports.daily import DailyTransport
from src.cmd.bots import register_ai_room_bots
from src.modules.speech.enhancer import SpeechEnhancerEnvInit


load_dotenv(override=True)


@register_ai_room_bots.register
class DailyAudioNoiseFilterBot(DailyRoomBot):
    """
    filter audio noise to echo
    - use daily audio stream(bytes) --> audio noise filter processor --> save
    """

    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()

        self.vad_analyzer = None
        self.audio_noise_filter = None
        self.se_args = {}

    def load(self):
        self.vad_analyzer = self.get_vad_analyzer()

        # SE engine
        se_config = self.bot_config().se
        self.se_args = {} if se_config.args is None else se_config.args
        self.audio_noise_filter = SpeechEnhancerEnvInit.initEngine(se_config.tag, **self.se_args)
        self.audio_noise_filter.warmup(self.session)

    async def arun(self):
        assert self.vad_analyzer is not None
        assert self.audio_noise_filter is not None

        self.params = DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=self.vad_analyzer,
            vad_audio_passthrough=True,
        )
        logging.info(f"params: {self.params}")

        transport = DailyTransport(
            self.args.room_url,
            self.args.token,
            self.args.bot_name,
            self.params,
        )

        audio_noise_filter = AudioNoiseFilter(
            se=self.audio_noise_filter, session=self.session, **self.se_args
        )
        user_audio_save_processor = None
        if self._save_audio:
            user_audio_save_processor = AudioSaveProcessor(
                prefix_name="user_speak", pass_raw_audio=True
            )

        self.task = PipelineTask(
            Pipeline(
                [
                    transport.input_processor(),
                    audio_noise_filter,
                    # UserAudioResponseAggregator(),
                    # user_audio_save_processor,
                    # FrameLogger(include_frame_types=[AudioRawFrame]),
                    transport.output_processor(),  # BotSpeakingFrame
                ]
            ),
            params=PipelineParams(
                allow_interruptions=False,
                enable_metrics=False,
                send_initial_empty_metrics=False,
            ),
        )

        transport.add_event_handler("on_first_participant_joined", self.on_first_participant_joined)
        transport.add_event_handler("on_participant_left", self.on_participant_left)
        transport.add_event_handler("on_call_state_updated", self.on_call_state_updated)

        await PipelineRunner().run(self.task)
