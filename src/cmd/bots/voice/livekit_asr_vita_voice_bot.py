import logging

from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.processors.logger import FrameLogger
from apipeline.frames import AudioRawFrame, TextFrame

from src.processors.aggregators.user_response import UserResponseAggregator
from src.cmd.bots.base_livekit import LivekitRoomBot
from src.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from src.common.types import LivekitParams
from src.transports.livekit import LivekitTransport
from src.cmd.bots import register_ai_room_bots

from dotenv import load_dotenv

load_dotenv(override=True)


@register_ai_room_bots.register
class LivekitAsrVITAVoiceBot(LivekitRoomBot):
    """
    use livekit audio stream(bytes) --> asr --> text ---> vita audio voice processor -->text/audio_bytes
    - don't support tools call, need sft
    """

    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()

    async def arun(self):
        self._vad_analyzer = VADAnalyzerEnvInit.initVADAnalyzerEngine()
        self.params = LivekitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=self._vad_analyzer,
            vad_audio_passthrough=True,
        )
        asr_processor = self.get_asr_processor()

        self._voice_processor = self.get_text_vita_voice_processor()
        stream_info = self._voice_processor.stream_info
        self.params.audio_out_sample_rate = stream_info["sample_rate"]
        self.params.audio_out_channels = stream_info["channels"]

        transport = LivekitTransport(
            self.args.token,
            params=self.params,
        )
        self.regisiter_room_event(transport)

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
        await PipelineRunner().run(self.task)

    async def on_first_participant_say_hi(self, transport: LivekitTransport, participant):
        pass
