import asyncio
from io import BytesIO
import logging
from typing import AsyncGenerator

from PIL import Image
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.runner import PipelineRunner
from apipeline.pipeline.task import PipelineTask, PipelineParams
from apipeline.frames.data_frames import Frame, TextFrame

from src.processors.user_image_request_processor import UserImageRequestProcessor
from src.common.utils.img_utils import image_bytes_to_base64_data_uri
from src.processors.vision.base import VisionProcessorBase
from src.processors.aggregators.user_response import UserResponseAggregator
from src.processors.aggregators.vision_image_frame import VisionImageFrameAggregator
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.common.types import DailyParams
from src.cmd.bots.base import DailyRoomBot
from src.transports.daily import DailyTransport
from src.types.frames.data_frames import VisionImageRawFrame
from .. import register_daily_room_bots


class MockVisionLMProcessor(VisionProcessorBase):
    """
    use vision lm to process image frames
    """

    def __init__(
        self,
    ):
        super().__init__()

    async def run_vision(self, frame: VisionImageRawFrame) -> AsyncGenerator[Frame, None]:
        logging.info(f"Mock Analyzing image: {frame}")

        image = Image.frombytes(frame.mode, frame.size, frame.image)
        with BytesIO() as buffered:
            image.save(buffered, format=frame.format)
            img_base64_str = image_bytes_to_base64_data_uri(
                buffered.getvalue(), frame.format.lower())

        logging.info(f"Mock img_base64_str+text: {img_base64_str} {frame.text}")

        # await asyncio.sleep(1)

        yield TextFrame(text=f"你好！niubility, 你他娘的是个人才。请访问 github achatbot 进行把玩, 可以在colab中部署免费把玩。")


@register_daily_room_bots.register
class DailyMockVisionBot(DailyRoomBot):
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

        # llm_in_aggr = LLMUserResponseAggregator()
        in_aggr = UserResponseAggregator()
        image_requester = UserImageRequestProcessor()
        vision_aggregator = VisionImageFrameAggregator()
        llm_processor = MockVisionLMProcessor()
        # llm_out_aggr = LLMAssistantResponseAggregator()

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport: DailyTransport, participant):
            transport.capture_participant_video(participant["id"], framerate=0)
            image_requester.set_participant_id(participant["id"])

        pipeline = Pipeline([
            transport.input_processor(),
            asr_processor,
            # llm_in_aggr,
            in_aggr,
            image_requester,
            vision_aggregator,
            llm_processor,
            tts_processor,
            transport.output_processor(),
            # llm_out_aggr,
        ])
        task = PipelineTask(pipeline, params=PipelineParams())
        await PipelineRunner().run(task)
