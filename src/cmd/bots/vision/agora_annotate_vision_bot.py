import logging

from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.parallel_pipeline import ParallelPipeline
from apipeline.pipeline.runner import PipelineRunner
from apipeline.pipeline.task import PipelineTask
from apipeline.processors.logger import FrameLogger

from src.processors.vision.annotate_processor import AnnotateProcessor
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.cmd.bots.base_agora import AgoraChannelBot
from src.common.types import AgoraParams
from src.transports.agora import AgoraTransport
from .. import register_ai_room_bots


@register_ai_room_bots.register
class AgoraAnnotateVisionBot(AgoraChannelBot):
    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()

    async def arun(self):
        agora_params = AgoraParams(
            audio_out_enabled=True,
            camera_in_enabled=True,
            camera_out_enabled=True,
            camera_out_is_live=True,
            camera_out_width=640,
            camera_out_height=480,
        )
        annotate_processor: AnnotateProcessor = self.get_vision_annotate_processor()
        tts_processor: TTSProcessor = self.get_tts_processor()
        stream_info = tts_processor.get_stream_info()
        agora_params.audio_out_sample_rate = stream_info["sample_rate"]
        agora_params.audio_out_channels = stream_info["channels"]
        transport = AgoraTransport(self.args.token, params=agora_params)

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(
            transport: AgoraTransport,
            user_id: str,
        ):
            # subscribed the first participant
            transport.capture_participant_video(user_id)

            participant_name = user_id
            await tts_processor.say(f"你好,{participant_name}。" f"这是一个图像检测注释demo。")

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
