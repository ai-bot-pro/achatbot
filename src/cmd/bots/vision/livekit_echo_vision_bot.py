import logging

from livekit import rtc
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.runner import PipelineRunner
from apipeline.pipeline.task import PipelineTask
from apipeline.processors.logger import FrameLogger

from src.cmd.bots.base_livekit import LivekitRoomBot
from src.common.types import LivekitParams
from src.types.frames.data_frames import UserImageRawFrame
from src.transports.livekit import LivekitTransport
from .. import register_ai_room_bots


@register_ai_room_bots.register
class LivekitEchoVisionBot(LivekitRoomBot):
    async def arun(self):
        transport = LivekitTransport(
            self.args.token,
            LivekitParams(
                camera_in_enabled=True,
                camera_out_enabled=True,
                camera_out_is_live=True,
                camera_out_width=1280,
                camera_out_height=720,
            ),
        )

        self.regisiter_room_event(transport)

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(
            transport: LivekitTransport,
            participant: rtc.RemoteParticipant,
        ):
            transport.capture_participant_video(participant.sid)

        @transport.event_handler("on_video_track_subscribed")
        async def on_video_track_subscribed(
            transport: LivekitTransport,
            participant: rtc.RemoteParticipant,
        ):
            transport.capture_participant_video(participant.sid)

        pipeline = Pipeline(
            [
                transport.input_processor(),
                # FrameLogger(include_frame_types=[UserImageRawFrame]),
                transport.output_processor(),
            ]
        )
        self.task = PipelineTask(pipeline)
        await PipelineRunner().run(self.task)
