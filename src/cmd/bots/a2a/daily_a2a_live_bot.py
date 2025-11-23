import logging
from typing import cast

from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.processors.logger import FrameLogger
from apipeline.frames import TextFrame

from src.modules.speech.turn_analyzer import TurnAnalyzerEnvInit
from src.common.types import DailyParams
from src.transports.daily import DailyTransport
from src.cmd.bots.base_daily import DailyRoomBot
from src.cmd.bots import register_ai_room_bots
from src.types.frames.control_frames import (
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from src.processors.a2a.a2a_live_processor import A2ALiveProcessor

from dotenv import load_dotenv

load_dotenv(override=True)


@register_ai_room_bots.register
class DailyA2ALiveBot(DailyRoomBot):
    """
    use a2a live bot
    - no VAD
    - text/images/audio -> gemini live (BIDI) -> text/audio
    """

    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()

    async def arun(self):
        self.daily_params = DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=False,
            vad_audio_passthrough=True,
            transcription_enabled=False,
            audio_out_sample_rate=24000,
        )

        transport = DailyTransport(
            self.args.room_url,
            self.args.token,
            self.args.bot_name,
            self.daily_params,
        )

        self.a2a_processor: A2ALiveProcessor = cast(
            A2ALiveProcessor,
            self.get_a2a_processor(tag="a2a_live_processor"),
        )

        self.task = PipelineTask(
            Pipeline(
                [
                    transport.input_processor(),
                    self.a2a_processor,
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

        await PipelineRunner(handle_sigint=self._handle_sigint).run(self.task)

    async def on_first_participant_say_hi(self, transport: DailyTransport, participant):
        self.a2a_processor.set_user_id(participant["id"])
        await self.a2a_processor.create_conversation()
        await self.a2a_processor.create_push_task()
        if self.daily_params.transcription_enabled:
            transport.capture_participant_transcription(participant["id"])

        is_cn = self._bot_config.a2a and self._bot_config.a2a.language == "zh"
        user_hi_text = "请用中文介绍下自己。" if is_cn else "Please introduce yourself first."
        await self.task.queue_frame(TextFrame(text=user_hi_text))
