import asyncio
from dataclasses import dataclass

import aiohttp
from apipeline.frames.sys_frames import Frame
from apipeline.pipeline.pipeline import Pipeline, FrameProcessor
from apipeline.pipeline.sync_parallel_pipeline import SyncParallelPipeline
from apipeline.pipeline.task import PipelineTask, PipelineParams
from apipeline.pipeline.runner import PipelineRunner
from apipeline.frames.data_frames import TextFrame, ImageRawFrame, AudioRawFrame
from apipeline.processors.frame_processor import FrameDirection
from apipeline.processors.aggregators.sentence import SentenceAggregator
from apipeline.processors.logger import FrameLogger

from src.processors.translation.google_translate_processor import GoogleTranslateProcessor
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.cmd.bots.base_daily import DailyRoomBot
from src.common.types import DailyParams
from src.types.frames.control_frames import LLMFullResponseStartFrame
from src.transports.daily import DailyTransport
from src.types.frames.data_frames import LLMMessagesFrame
from .. import register_ai_room_bots


@dataclass
class MonthFrame(Frame):
    month: str

    def __str__(self):
        return f"{self.name}(month: {self.month})"


class MonthPrepender(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.most_recent_month = ""
        self.prepend_to_next_text_frame = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, MonthFrame):
            self.most_recent_month = frame.month
        elif self.prepend_to_next_text_frame and isinstance(frame, TextFrame):
            await self.push_frame(TextFrame(f"{self.most_recent_month}: {frame.text}"))
            self.prepend_to_next_text_frame = False
        elif isinstance(frame, LLMFullResponseStartFrame):
            self.prepend_to_next_text_frame = True
            await self.push_frame(frame)
        else:
            await self.push_frame(frame, direction)


@register_ai_room_bots.register
class DailyMonthNarrationBot(DailyRoomBot):
    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()

    async def arun(self):
        async with aiohttp.ClientSession() as session:
            daily_params = DailyParams(
                audio_out_enabled=True,
                camera_out_enabled=True,
                camera_out_width=1280,
                camera_out_height=720,
            )

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
            transport.add_event_handler(
                "on_first_participant_joined", self.on_first_participant_joined
            )
            transport.add_event_handler("on_participant_left", self.on_participant_left)
            transport.add_event_handler("on_call_state_updated", self.on_call_state_updated)

            llm_processor = self.get_llm_processor()
            sentence_aggregator = SentenceAggregator()
            month_prepender = MonthPrepender()

            image_gen_processor = self.get_image_gen_processor()
            image_gen_processor.set_aiohttp_session(session)
            image_gen_processor.set_size(
                width=daily_params.camera_out_width,
                height=daily_params.camera_out_height,
            )

            translate_processor = GoogleTranslateProcessor()

            frames = []
            for month in [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ]:
                messages = [
                    {
                        "role": "system",
                        "content": f"Describe a nature photograph suitable for use in a calendar, for the month of {month}. Include only the image description with no preamble. Limit the description to one sentence, please.",
                    }
                ]
                frames.append(MonthFrame(month=month))
                frames.append(LLMMessagesFrame(messages))

            self.task = PipelineTask(
                Pipeline(
                    [
                        llm_processor,  # LLM
                        sentence_aggregator,  # Aggregates LLM output into full sentences
                        SyncParallelPipeline(  # Run pipelines in parallel aggregating the result
                            # Create "Month: sentence" and output audio
                            [month_prepender, translate_processor, tts_processor],
                            [image_gen_processor],  # Generate image
                        ),
                        FrameLogger(include_frame_types=[ImageRawFrame, AudioRawFrame]),
                        transport.output_processor(),
                    ]
                )
            )
            await self.task.queue_frames(frames)
            await PipelineRunner().run(self.task)
