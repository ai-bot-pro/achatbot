import logging

from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.processors.filters.null_filter import NullFilter
from apipeline.processors.filters.function_filter import FunctionFilter
from apipeline.processors.filters.frame_filter import FrameFilter
from apipeline.pipeline.runner import PipelineRunner
from apipeline.pipeline.parallel_pipeline import ParallelPipeline
from apipeline.processors.logger import FrameLogger
from apipeline.frames import AudioRawFrame, TextFrame

from src.processors.speech.audio_save_processor import AudioSaveProcessor
from src.processors.aggregators.user_audio_response import UserAudioResponseAggregator
from src.cmd.bots.base_daily import DailyRoomBot
from src.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from src.common.types import DailyParams
from src.transports.daily import DailyTransport
from src.cmd.bots import register_ai_room_bots
from src.types.frames import *

from dotenv import load_dotenv

load_dotenv(override=True)


@register_ai_room_bots.register
class DailyFreezeOmniVoiceBot(DailyRoomBot):
    """
    use daily audio stream(bytes) --> FreezeOmni voice processor -->text/audio_bytes
    - Tech Report: https://arxiv.org/pdf/2411.00774
    """

    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()

    async def arun(self):
        self._vad_analyzer = VADAnalyzerEnvInit.initVADAnalyzerEngine()
        self.params = DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=self._vad_analyzer,
            vad_audio_passthrough=True,
        )

        self._voice_processor = self.get_audio_freeze_omni_voice_processor()
        stream_info = self._voice_processor.stream_info
        self.params.audio_out_sample_rate = stream_info["sample_rate"]
        self.params.audio_out_channels = stream_info["channels"]

        transport = DailyTransport(
            self.args.room_url,
            self.args.token,
            self.args.bot_name,
            self.params,
        )

        # messages = []
        # if self._bot_config.llm.messages:
        #     messages = self._bot_config.llm.messages

        _ = ParallelPipeline(
            [  # save user audio pipeline
                UserAudioResponseAggregator(),
                FrameLogger(
                    prefix="user audio aggr to save ==>",
                    include_frame_types=[AudioRawFrame],
                ),
                AudioSaveProcessor(prefix_name="user_audio_aggr"),
                FrameLogger(include_frame_types=[PathAudioRawFrame]),
                NullFilter(),
                # check no downstream frame to log
                # FrameLogger(),
            ],
            [  # bot speak and save pipeline
                UserAudioResponseAggregator(),
                FrameLogger(
                    prefix="user audio aggr ==>",
                    include_frame_types=[AudioRawFrame],
                ),
                self._voice_processor,
                FrameLogger(
                    prefix="bot audio speak ==>",
                    include_frame_types=[AudioRawFrame, TextFrame],
                ),
                AudioSaveProcessor(prefix_name="bot_speak"),
                FrameLogger(
                    prefix="bot speak to save ==>",
                    include_frame_types=[PathAudioRawFrame],
                ),
            ],
        )

        single_pipeline = Pipeline(
            [
                UserAudioResponseAggregator(),
                FrameLogger(
                    prefix="user audio aggr ==>",
                    include_frame_types=[AudioRawFrame],
                ),
                self._voice_processor,
                FrameLogger(
                    prefix="bot audio speak ==>",
                    include_frame_types=[AudioRawFrame, TextFrame],
                ),
            ]
        )

        self.task = PipelineTask(
            Pipeline(
                [
                    transport.input_processor(),
                    # parallel_pipeline,
                    single_pipeline,
                    transport.output_processor(),
                ]
            ),
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

        await PipelineRunner().run(self.task)

    async def on_first_participant_say_hi(self, transport: DailyTransport, participant):
        pass
