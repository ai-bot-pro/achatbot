import logging

from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.runner import PipelineRunner
from apipeline.pipeline.task import PipelineTask
from apipeline.processors.logger import FrameLogger

from src.cmd.bots.base_agora import AgoraChannelBot
from src.common.types import AgoraParams
from src.types.frames.data_frames import UserImageRawFrame
from src.transports.agora import AgoraTransport
from .. import register_ai_room_bots


@register_ai_room_bots.register
class AgoraEchoVisionBot(AgoraChannelBot):
    async def arun(self):
        transport = AgoraTransport(
            self.args.token,
            AgoraParams(
                camera_in_enabled=True,
                camera_out_enabled=True,
                camera_out_is_live=True,
                camera_out_width=640,
                camera_out_height=480,
            ),
        )

        self.regisiter_room_event(transport)

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(
            transport: AgoraTransport,
            user_id: str,
        ):
            logging.info(f"fisrt joined user_id:{user_id}")
            transport.capture_participant_video(user_id)

        pipeline = Pipeline(
            [
                transport.input_processor(),
                # FrameLogger(include_frame_types=[UserImageRawFrame]),
                transport.output_processor(),
            ]
        )
        self.task = PipelineTask(pipeline)
        await PipelineRunner().run(self.task)
