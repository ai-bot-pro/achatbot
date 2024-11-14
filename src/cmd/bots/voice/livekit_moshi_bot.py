import logging

from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.processors.logger import FrameLogger
from apipeline.frames import AudioRawFrame, TextFrame

from src.processors.voice.moshi_voice_processor import MoshiVoiceProcessor
from src.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from src.common.types import LivekitParams
from src.transports.livekit import LivekitTransport
from src.cmd.bots.base_livekit import LivekitRoomBot
from src.cmd.bots import register_ai_room_bots
from src.types.frames.data_frames import LLMMessagesFrame
from src.types.llm.lmgen import LMGenArgs

from dotenv import load_dotenv
load_dotenv(override=True)


@register_ai_room_bots.register
class LivekitMoshiVoiceBot(LivekitRoomBot):
    """
    use moshi voice processor, just a simple chat bot.
    """

    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()

    async def arun(self):
        vad_analyzer = VADAnalyzerEnvInit.initVADAnalyzerEngine()
        self.params = LivekitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=vad_analyzer,
            vad_audio_passthrough=True,
        )

        voice_processor = MoshiVoiceProcessor(lm_gen_args=LMGenArgs())
        stream_info = voice_processor.stream_info
        self.params.audio_out_sample_rate = stream_info["sample_rate"]
        self.params.audio_out_channels = stream_info["channels"]

        transport = LivekitTransport(
            self.args.token,
            params=self.params,
        )

        messages = []
        if self._bot_config.llm.messages:
            messages = self._bot_config.llm.messages

        self.task = PipelineTask(
            Pipeline([
                transport.input_processor(),
                FrameLogger(include_frame_types=[AudioRawFrame]),
                voice_processor,
                FrameLogger(include_frame_types=[AudioRawFrame, TextFrame]),
                transport.output_processor(),
            ]),
            params=PipelineParams(
                allow_interruptions=False,
                enable_metrics=True,
                send_initial_empty_metrics=False,
            ),
        )

        transport.add_event_handlers(
            "on_first_participant_joined",
            [self.on_first_participant_say_hi])

        await PipelineRunner().run(self.task)

    async def on_first_participant_say_hi(self, transport: LivekitTransport, participant):
        pass