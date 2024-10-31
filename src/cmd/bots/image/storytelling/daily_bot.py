import asyncio
from dataclasses import dataclass

import aiohttp
from apipeline.frames.sys_frames import Frame
from apipeline.pipeline.pipeline import Pipeline, FrameProcessor
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.frames.data_frames import TextFrame, ImageRawFrame, AudioRawFrame
from apipeline.processors.frame_processor import FrameDirection
from apipeline.processors.aggregators.sentence import SentenceAggregator
from apipeline.processors.logger import FrameLogger

from src.processors.aggregators.llm_response import LLMAssistantResponseAggregator, LLMUserResponseAggregator
from src.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.cmd.bots.base_daily import DailyRoomBot
from src.common.types import DailyParams
from src.types.frames.control_frames import LLMFullResponseStartFrame
from src.transports.daily import DailyTransport
from src.types.frames.data_frames import DailyTransportMessageFrame, LLMMessagesFrame
from src.cmd.bots import register_ai_room_bots


from .utils.helpers import load_images, load_sounds
from .prompts import CUE_USER_TURN, LLM_INTRO_PROMPT


sounds = load_sounds(["listening.wav"])
images = load_images(["book1.png", "book2.png"])


@register_ai_room_bots.register
class DailyStoryTellingBot(DailyRoomBot):
    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()

    async def arun(self):
        async with aiohttp.ClientSession() as session:
            vad_analyzer = VADAnalyzerEnvInit.initVADAnalyzerEngine()
            self.daily_params = DailyParams(
                audio_in_enabled=True,
                vad_enabled=True,
                vad_analyzer=vad_analyzer,
                vad_audio_passthrough=True,
                transcription_enabled=False,
                audio_out_enabled=True,
                camera_out_enabled=True,
                camera_out_width=1280,
                camera_out_height=720
            )

            asr_processor = self.get_asr_processor()
            llm_processor = self.get_llm_processor()
            messages = []
            if self._bot_config.llm.messages:
                messages = self._bot_config.llm.messages
            user_response = LLMUserResponseAggregator(messages)
            assistant_response = LLMAssistantResponseAggregator(messages)

            tts_processor: TTSProcessor = self.get_tts_processor()
            stream_info = tts_processor.get_stream_info()
            self.daily_params.audio_out_sample_rate = stream_info["sample_rate"]
            self.daily_params.audio_out_channels = stream_info["channels"]

            image_gen_processor = self.get_image_gen_processor()
            image_gen_processor.set_aiohttp_session(session)
            image_gen_processor.set_size(
                width=self.daily_params.camera_out_width,
                height=self.daily_params.camera_out_height,
            )

            transport = DailyTransport(
                self.args.room_url, self.args.token, self.args.bot_name,
                self.daily_params,
            )
            transport.add_event_handler(
                "on_first_participant_joined",
                [
                    self.on_first_participant_joined,
                    self.on_first_participant_say_hi,
                ]
            )
            transport.add_event_handler(
                "on_participant_left",
                self.on_participant_left)
            transport.add_event_handler(
                "on_call_state_updated",
                self.on_call_state_updated)

            self.task = PipelineTask(
                Pipeline([
                    transport.input_processor(),
                    asr_processor,
                    user_response,
                    llm_processor,
                    image_gen_processor,
                    tts_processor,
                    transport.output_processor(),
                    assistant_response,
                ]),
                params=PipelineParams(
                    allow_interruptions=True,
                    enable_metrics=True,
                    send_initial_empty_metrics=False,
                ),
            )

            await PipelineRunner().run(self.task)

    async def on_first_participant_say_hi(self, transport: DailyTransport, participant):
        if self.daily_params.transcription_enabled:
            transport.capture_participant_transcription(participant["id"])

        # joined use tts say "hello" to introduce with llm generate
        if self._bot_config.tts \
                and self._bot_config.llm \
                and self._bot_config.llm.messages \
                and len(self._bot_config.llm.messages) == 1:
            hi_text = "Please introduce yourself first."
            if self._bot_config.llm.language \
                    and self._bot_config.llm.language == "zh":
                hi_text = "请用中文介绍下自己。"
            self._bot_config.llm.messages.append({
                "role": "user",
                "content": hi_text,
            })
            await self.task.queue_frames([
                #LLMMessagesFrame(self._bot_config.llm.messages),
                images["book1"],
                LLMMessagesFrame([LLM_INTRO_PROMPT]),
                DailyTransportMessageFrame(CUE_USER_TURN),
                sounds["listening"],
                images["book2"],
            ])
