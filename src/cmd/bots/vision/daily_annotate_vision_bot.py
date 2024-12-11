from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.parallel_pipeline import ParallelPipeline
from apipeline.pipeline.runner import PipelineRunner
from apipeline.pipeline.task import PipelineTask
from apipeline.processors.logger import FrameLogger

from src.processors.vision.annotate_processor import AnnotateProcessor
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.common.types import DailyParams
from src.cmd.bots.base_daily import DailyRoomBot
from src.transports.daily import DailyTransport
from .. import register_ai_room_bots


@register_ai_room_bots.register
class DailyAnnotateVisionBot(DailyRoomBot):
    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()

    async def arun(self):
        daily_params = DailyParams(
            audio_out_enabled=True,
            camera_out_enabled=True,
            camera_out_is_live=True,
            camera_out_width=1280,
            camera_out_height=720,
        )
        self.annotate_processor: AnnotateProcessor = self.get_vision_annotate_processor()
        self.tts_processor: TTSProcessor = self.get_tts_processor()
        stream_info = self.tts_processor.get_stream_info()

        daily_params.audio_out_sample_rate = stream_info["sample_rate"]
        daily_params.audio_out_channels = stream_info["channels"]
        transport = DailyTransport(
            self.args.room_url, self.args.token, self.args.bot_name, daily_params
        )

        transport.add_event_handler("on_first_participant_joined", self.on_first_participant_joined)
        transport.add_event_handler("on_participant_left", self.on_participant_left)
        transport.add_event_handler("on_call_state_updated", self.on_call_state_updated)

        pipeline = Pipeline(
            [
                transport.input_processor(),
                ParallelPipeline(
                    [self.annotate_processor],
                    [self.tts_processor],
                ),
                # FrameLogger(include_frame_types=[UserImageRawFrame]),
                transport.output_processor(),
            ]
        )
        self.task = PipelineTask(pipeline)
        await PipelineRunner().run(self.task)

    async def on_first_participant_joined(self, transport: DailyTransport, participant):
        transport.capture_participant_video(participant["id"])
        await self.tts_processor.say("你好。这是一个图像检测注释demo。")
