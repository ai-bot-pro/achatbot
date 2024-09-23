import logging

from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.parallel_pipeline import ParallelPipeline
from apipeline.pipeline.runner import PipelineRunner
from apipeline.processors.filters.function_filter import FunctionFilter
from apipeline.pipeline.task import PipelineTask, PipelineParams
from apipeline.frames.data_frames import Frame, TextFrame, ImageRawFrame

from src.processors.aggregators.llm_response import LLMAssistantResponseAggregator, LLMUserResponseAggregator
from src.processors.user_image_request_processor import UserImageRequestProcessor
from src.processors.aggregators.user_response import UserResponseAggregator
from src.processors.aggregators.vision_image_frame import VisionImageFrameAggregator
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.common.types import DailyParams
from src.cmd.bots.base import DailyRoomBot
from src.transports.daily import DailyTransport
from .. import register_daily_room_bots


@register_daily_room_bots.register
class DailyChatVisionBot(DailyRoomBot):
    """
    use gen text llm model to chat
    when gen text is  about describe the image, use vision model to describe the image with describe text
    if gen text is about describe the image, and image; filter text and image
    """

    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()

    async def text_filter(self, frame: Frame):
        if isinstance(frame, TextFrame):
            if frame.text == self.text:
                return False
        return True

    async def image_filter(self, frame: Frame):
        if isinstance(frame, ImageRawFrame):
            return False
        return True

    async def arun(self):
        vad_analyzer = self.get_vad_analyzer()
        daily_params = DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=vad_analyzer,
            vad_audio_passthrough=True,
            transcription_enabled=False,
        )

        asr_processor = self.get_asr_processor()

        tts_processor: TTSProcessor = self.get_tts_processor()
        stream_info = tts_processor.get_stream_info()

        daily_params.audio_out_sample_rate = stream_info["sample_rate"]
        daily_params.audio_out_channels = stream_info["channels"]
        transport = DailyTransport(
            self.args.room_url, self.args.token, self.args.bot_name,
            daily_params,
        )

        llm_in_aggr = LLMUserResponseAggregator()
        llm_processor = self.get_openai_llm_processor()

        in_aggr = UserResponseAggregator()
        image_requester = UserImageRequestProcessor()
        vision_aggregator = VisionImageFrameAggregator()
        vision_llm_processor = self.get_vision_llm_processor()

        text_filter = FunctionFilter(filter=self.text_filter),
        img_filter = FunctionFilter(filter=self.image_filter),

        llm_out_aggr = LLMAssistantResponseAggregator()

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport: DailyTransport, participant):
            transport.capture_participant_video(participant["id"], framerate=0)
            image_requester.set_participant_id(participant["id"])

        pipeline = Pipeline([
            transport.input_processor(),
            asr_processor,
            llm_in_aggr,
            llm_processor,
            ParallelPipeline(
                [in_aggr, image_requester, vision_aggregator, vision_llm_processor],
                [text_filter, img_filter],
            ),
            tts_processor,
            transport.output_processor(),
            llm_out_aggr,
        ])
        task = PipelineTask(pipeline, params=PipelineParams())
        await PipelineRunner().run(task)
