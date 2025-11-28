from typing import cast

from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.frames import TextFrame, AudioRawFrame, ImageRawFrame
from apipeline.processors.logger import FrameLogger

from src.common.types import DailyParams, VADState
from src.transports.daily import DailyTransport
from src.cmd.bots.base_daily import DailyRoomBot
from src.cmd.bots import register_ai_room_bots
from src.processors.a2a.a2a_live_processor import A2ALiveProcessor
from src.processors.user_image_request_processor import UserVADImageRequestProcessor


from dotenv import load_dotenv

load_dotenv(override=True)


@register_ai_room_bots.register
class DailyVADA2ALiveBot(DailyRoomBot):
    """
    use a2a live bot
    - text/images/audio -> vad + gemini live (BIDI) -> text/audio
    """

    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()

    def load(self):
        self.vad_analyzer_pool = self.get_vad_analyzer_pool()

    async def arun(self):
        self.daily_params = DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_audio_passthrough=True,
            transcription_enabled=False,
            audio_out_sample_rate=24000,
        )
        if self.vad_analyzer_pool:
            vad_analyzer_info, vad_analyzer = self.get_vad_analyzer_from_pool()
            self.daily_params.vad_analyzer = vad_analyzer

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

        self.image_requester = UserVADImageRequestProcessor(
            states=[VADState.SPEAKING],
            interval_time_ms=self._bot_config.a2a.interval_time_ms,
        )

        pipeline = Pipeline(
            [  # audio->text/audio BIDI
                transport.input_processor(),
                self.a2a_processor,
                transport.output_processor(),
            ]
        )
        if self.vad_analyzer_pool:
            pipeline = Pipeline(
                [  # active images + audio -> text/audio BIDI
                    transport.input_processor(),
                    self.image_requester,
                    FrameLogger(include_frame_types=[ImageRawFrame]),
                    self.a2a_processor,
                    transport.output_processor(),
                ]
            )

        self.task = PipelineTask(
            pipeline,
            params=self._pipeline_params,
        )

        transport.add_event_handlers(
            "on_first_participant_joined",
            [self.on_first_participant_joined, self.on_first_participant_say_hi],
        )
        transport.add_event_handler("on_participant_left", self.on_participant_left)
        transport.add_event_handler("on_call_state_updated", self.on_call_state_updated)

        await PipelineRunner(handle_sigint=self._handle_sigint).run(self.task)
        # Ensure instances are returned to the pool
        if self.vad_analyzer_pool and vad_analyzer_info:
            self.vad_analyzer_pool.put(vad_analyzer_info)

    async def on_first_participant_say_hi(self, transport: DailyTransport, participant):
        transport.capture_participant_video(participant["id"], framerate=0)
        self.image_requester.set_participant_id(participant["id"])

        self.a2a_processor.set_user_id(participant["id"])
        await self.a2a_processor.create_conversation()
        await self.a2a_processor.create_push_task()

        is_cn = self._bot_config.a2a and self._bot_config.a2a.language == "zh"
        user_hi_text = "请用中文介绍下自己。" if is_cn else "Please introduce yourself first."
        await self.task.queue_frame(TextFrame(text=user_hi_text))
