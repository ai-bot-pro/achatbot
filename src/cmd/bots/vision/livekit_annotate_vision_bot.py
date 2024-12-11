from livekit import rtc
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.parallel_pipeline import ParallelPipeline
from apipeline.pipeline.runner import PipelineRunner
from apipeline.pipeline.task import PipelineTask
from apipeline.processors.logger import FrameLogger

from src.processors.vision.annotate_processor import AnnotateProcessor
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.cmd.bots.base_livekit import LivekitRoomBot
from src.common.types import LivekitParams
from src.transports.livekit import LivekitTransport
from .. import register_ai_room_bots


@register_ai_room_bots.register
class LivekitAnnotateVisionBot(LivekitRoomBot):
    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()

    async def arun(self):
        livekit_params = LivekitParams(
            audio_out_enabled=True,
            camera_in_enabled=True,
            camera_out_enabled=True,
            camera_out_is_live=True,
            camera_out_width=1280,
            camera_out_height=720,
        )
        annotate_processor: AnnotateProcessor = self.get_vision_annotate_processor()
        tts_processor: TTSProcessor = self.get_tts_processor()
        stream_info = tts_processor.get_stream_info()
        livekit_params.audio_out_sample_rate = stream_info["sample_rate"]
        livekit_params.audio_out_channels = stream_info["channels"]
        transport = LivekitTransport(self.args.token, params=livekit_params)

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(
            transport: LivekitTransport,
            participant: rtc.RemoteParticipant,
        ):
            # subscribed the first participant
            transport.capture_participant_video(participant.sid)

            participant_name = participant.name if participant.name else participant.identity
            await tts_processor.say(f"你好,{participant_name}。" f"这是一个图像检测注释demo。")

        @transport.event_handler("on_video_track_subscribed")
        async def on_video_track_subscribed(
            transport: LivekitTransport,
            participant: rtc.RemoteParticipant,
        ):
            transport.capture_participant_video(participant.sid)

        pipeline = Pipeline(
            [
                transport.input_processor(),
                ParallelPipeline(
                    [annotate_processor],
                    [tts_processor],
                ),
                # FrameLogger(include_frame_types=[UserImageRawFrame]),
                transport.output_processor(),
            ]
        )
        self.task = PipelineTask(pipeline)
        await PipelineRunner().run(self.task)
