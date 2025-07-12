import asyncio
import logging

from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.runner import PipelineRunner
from apipeline.pipeline.task import PipelineTask, PipelineParams

from src.processors.user_image_request_processor import UserImageRequestProcessor
from src.processors.aggregators.user_response import UserResponseAggregator
from src.processors.aggregators.vision_image_frame import VisionImageFrameAggregator
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.common.types import DailyParams
from src.cmd.bots.base_daily import DailyRoomBot
from src.transports.daily import DailyTransport
from .. import register_ai_room_bots


@register_ai_room_bots.register
class DailyDescribeVisionBot(DailyRoomBot):
    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()
        self.llm_engine = None

    def load(self):
        from src.core.llm import LLMEnvInit

        try:
            vision_llm_config = self._bot_config.llm or self._bot_config.vision_llm
            if "fastdeploy" in vision_llm_config.tag:
                self.llm_engine = LLMEnvInit.initLLMEngine(
                    vision_llm_config.tag, vision_llm_config.args
                )
        except Exception as e:
            logging.error(f"err: {e}", exc_info=True)
            raise e

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
            self.args.room_url,
            self.args.token,
            self.args.bot_name,
            daily_params,
        )

        # llm_in_aggr = LLMUserResponseAggregator()
        in_aggr = UserResponseAggregator()
        image_requester = UserImageRequestProcessor()
        vision_aggregator = VisionImageFrameAggregator()
        llm_processor = self.get_llm_processor(llm_engine=self.llm_engine)
        # llm_out_aggr = LLMAssistantResponseAggregator()

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport: DailyTransport, participant):
            transport.capture_participant_video(participant["id"], framerate=0)
            image_requester.set_participant_id(participant["id"])
            await asyncio.sleep(3)
            await tts_processor.say(
                "你好，欢迎使用 Vision Bot. 我是一名虚拟助手，可以结合视频进行提问。"
            )

        transport.add_event_handler("on_participant_left", self.on_participant_left)
        transport.add_event_handler("on_call_state_updated", self.on_call_state_updated)

        pipeline = Pipeline(
            [
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
            ]
        )
        self.task = PipelineTask(pipeline, params=PipelineParams())
        await PipelineRunner().run(self.task)
