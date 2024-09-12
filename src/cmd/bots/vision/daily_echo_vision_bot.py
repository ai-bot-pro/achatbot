from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.runner import PipelineRunner
from apipeline.pipeline.task import PipelineTask

from src.common.types import DailyParams
from src.cmd.bots.base import DailyRoomBot
from src.transports.daily import DailyTransport
from .. import register_daily_room_bots


@register_daily_room_bots.register
class DailyEchoVisionBot(DailyRoomBot):
    async def arun(self):
        transport = DailyTransport(
            self.args.room_url, self.args.token, self.args.bot_name,
            DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                camera_out_enabled=True,
                camera_out_is_live=True,
                camera_out_width=1280,
                camera_out_height=720
            )
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            transport.capture_participant_video(participant["id"])

        pipeline = Pipeline([
            transport.input_processor(),
            transport.output_processor(),
        ])
        task = PipelineTask(pipeline)
        await PipelineRunner().run(task)
