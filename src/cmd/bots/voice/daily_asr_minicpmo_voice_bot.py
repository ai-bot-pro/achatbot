import logging

from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.processors.logger import FrameLogger
from apipeline.frames import AudioRawFrame, TextFrame

from src.processors.aggregators.user_response import UserResponseAggregator
from src.cmd.bots.base_daily import DailyRoomBot
from src.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from src.common.types import DailyParams
from src.transports.daily import DailyTransport
from src.cmd.bots import register_ai_room_bots

from dotenv import load_dotenv

load_dotenv(override=True)


@register_ai_room_bots.register
class DailyAsrMiniCPMoVoiceBot(DailyRoomBot):
    """
    use daily audio stream(bytes) --> asr --> text ---> MiniCPMo voice processor -->text/audio_bytes
    - if use tools, need ft base model MiniCPM-o 2.6 with audio tokenizer, use tools instruct dataset, see: https://github.com/OpenBMB/MiniCPM-o?tab=readme-ov-file#fine-tuning
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
        asr_processor = self.get_asr_processor()

        self._voice_processor = self.get_text_minicpmo_voice_processor()
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
        #    messages = self._bot_config.llm.messages

        self.task = PipelineTask(
            Pipeline(
                [
                    transport.input_processor(),
                    asr_processor,
                    UserResponseAggregator(),
                    FrameLogger(include_frame_types=[TextFrame]),
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

        transport.add_event_handlers(
            "on_first_participant_joined",
            [self.on_first_participant_joined, self.on_first_participant_say_hi],
        )
        transport.add_event_handler("on_participant_left", self.on_participant_left)
        transport.add_event_handler("on_call_state_updated", self.on_call_state_updated)

        await PipelineRunner().run(self.task)

    async def on_first_participant_say_hi(self, transport: DailyTransport, participant):
        pass
