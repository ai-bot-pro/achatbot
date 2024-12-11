from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.runner import PipelineRunner
from apipeline.pipeline.task import PipelineTask
from apipeline.processors.logger import FrameLogger

from src.common.types import DailyParams
from src.cmd.bots.base_daily import DailyRoomBot
from src.transports.daily import DailyTransport
from src.types.frames.data_frames import UserImageRawFrame
from .. import register_ai_room_bots


@register_ai_room_bots.register
class DailyEchoVisionBot(DailyRoomBot):
    async def arun(self):
        transport = DailyTransport(
            self.args.room_url,
            self.args.token,
            self.args.bot_name,
            DailyParams(
                audio_out_enabled=True,
                camera_out_enabled=True,
                camera_out_is_live=True,
                camera_out_width=1280,
                camera_out_height=720,
            ),
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            transport.capture_participant_video(participant["id"])

        transport.add_event_handler("on_participant_left", self.on_participant_left)
        transport.add_event_handler("on_call_state_updated", self.on_call_state_updated)

        pipeline = Pipeline(
            [
                transport.input_processor(),
                # FrameLogger(include_frame_types=[UserImageRawFrame]),
                transport.output_processor(),
            ]
        )
        self.task = PipelineTask(pipeline)
        await PipelineRunner().run(self.task)
