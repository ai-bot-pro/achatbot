from apipeline.pipeline.pipeline import Pipeline, FrameProcessor
from apipeline.pipeline.runner import PipelineRunner
from apipeline.pipeline.task import PipelineTask, FrameDirection, PipelineParams
from apipeline.frames.data_frames import Frame, TextFrame

from src.processors.aggregators.llm_response import LLMAssistantResponseAggregator, LLMUserResponseAggregator
from src.processors.aggregators.vision_image_frame import VisionImageFrameAggregator
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.common.types import DailyParams
from src.cmd.bots.base import DailyRoomBot
from src.transports.daily import DailyTransport
from src.types.frames.control_frames import UserImageRequestFrame
from types.frames.data_frames import UserImageRawFrame
from .. import register_daily_room_bots


class UserImageRequester(FrameProcessor):
    def __init__(
            self,
            participant_id: str | None = None,
            init_user_prompt: str | list = "let me take a look",
    ):
        super().__init__()
        self._participant_id = participant_id
        self._init_user_prompt = init_user_prompt

    def set_participant_id(self, participant_id: str):
        self._participant_id = participant_id

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if self._participant_id and isinstance(frame, TextFrame):
            await self.push_frame(UserImageRequestFrame(self._participant_id), FrameDirection.UPSTREAM)
        await self.push_frame(frame, direction)


@register_daily_room_bots.register
class DailyDescribeVisionBot(DailyRoomBot):
    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()

    async def arun(self):
        vad_analyzer = self.get_vad_analyzer()
        daily_params = DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=vad_analyzer,
            vad_audio_passthrough=True,
            transcription_enabled=False,
            camera_out_enabled=True,
            camera_out_is_live=True,
            camera_out_width=1280,
            camera_out_height=720
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
        image_requester = UserImageRequester()
        vision_aggregator = VisionImageFrameAggregator()
        llm_processor = self.get_llm_processor()
        llm_out_aggr = LLMAssistantResponseAggregator()

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport: DailyTransport, participant):
            transport.capture_participant_video(participant["id"])
            transport.capture_participant_transcription(participant["id"])
            image_requester.set_participant_id(participant["id"])

        pipeline = Pipeline([
            transport.input_processor(),
            asr_processor,
            llm_in_aggr,
            image_requester,
            vision_aggregator,
            llm_processor,
            tts_processor,
            transport.output_processor(),
            llm_out_aggr,
        ])
        task = PipelineTask(pipeline, params=PipelineParams())
        await PipelineRunner().run(task)
