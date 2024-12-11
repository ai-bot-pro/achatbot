import copy

import aiohttp
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.frames.data_frames import TextFrame, ImageRawFrame, AudioRawFrame
from apipeline.frames.sys_frames import StopTaskFrame
from apipeline.processors.logger import FrameLogger

from src.processors.app_message_processor import (
    AppMessageControllProcessor,
    BotLLMTextProcessor,
    BotTTSTextProcessor,
    UserTranscriptionProcessor,
)
from src.processors.tranlate.google_translate_processor import GoogleTranslateProcessor
from src.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator,
)
from src.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.cmd.bots.base_daily import DailyRoomBot
from src.common.types import DailyParams
from src.transports.daily import DailyTransport
from src.types.frames.data_frames import DailyTransportMessageFrame, LLMMessagesFrame
from src.cmd.bots import register_ai_room_bots
from src.types.speech.language import TO_LLM_LANGUAGE


from .utils.helpers import load_images, load_sounds
from .processors import StoryImageFrame, StoryPageFrame, StoryProcessor, StoryPromptFrame
from .prompts import CUE_USER_TURN, LLM_BASE_PROMPT, LLM_INTRO_PROMPT


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
                camera_out_width=768,
                camera_out_height=768,
            )

            asr_processor = self.get_asr_processor()
            llm_processor = self.get_llm_processor()

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
            image_gen_processor.set_gen_image_frame(StoryImageFrame)

            transport = DailyTransport(
                self.args.room_url,
                self.args.token,
                self.args.bot_name,
                self.daily_params,
            )
            transport.add_event_handlers(
                "on_first_participant_joined",
                [
                    self.on_first_participant_joined,
                    self.on_first_participant_say_hi,
                ],
            )
            transport.add_event_handler("on_participant_left", self.on_participant_left)
            transport.add_event_handler("on_call_state_updated", self.on_call_state_updated)

            self.intro_task = PipelineTask(
                Pipeline(
                    [
                        llm_processor,
                        # FrameLogger(include_frame_types=[TextFrame]),
                        tts_processor,
                        AppMessageControllProcessor(),
                        BotTTSTextProcessor(),
                        transport.output_processor(),
                    ]
                ),
            )
            self.runner = PipelineRunner()
            # run the intro pipeline, task will exit after StopTaskFrame is processed.
            await self.runner.run(self.intro_task)

            language = self._bot_config.llm.language if self._bot_config.llm.language else "en"
            translate_processor = GoogleTranslateProcessor(src=language, target="en")
            translate_processor.set_translate_frame(StoryImageFrame)

            story_pages = []
            story_processor = StoryProcessor(self._bot_config.llm.messages, story_pages)
            user_response = LLMUserResponseAggregator(self._bot_config.llm.messages)
            assistant_response = LLMAssistantResponseAggregator(self._bot_config.llm.messages)

            self.task = PipelineTask(
                Pipeline(
                    [
                        transport.input_processor(),
                        asr_processor,
                        AppMessageControllProcessor(),
                        UserTranscriptionProcessor(),
                        user_response,
                        llm_processor,
                        # BotLLMTextProcessor(),
                        # FrameLogger(include_frame_types=[TextFrame]),
                        story_processor,
                        # FrameLogger(
                        #    include_frame_types=[
                        #        StoryImageFrame,
                        #        StoryPageFrame,
                        #        StoryPromptFrame,
                        #    ]
                        # ),
                        translate_processor,
                        FrameLogger(include_frame_types=[StoryImageFrame]),
                        image_gen_processor,
                        tts_processor,
                        # FrameLogger(include_frame_types=[ImageRawFrame, AudioRawFrame]),
                        AppMessageControllProcessor(),
                        BotTTSTextProcessor(),
                        transport.output_processor(),
                        assistant_response,
                    ]
                ),
                params=PipelineParams(
                    allow_interruptions=False,
                    enable_metrics=True,
                    send_initial_empty_metrics=False,
                ),
            )

            await self.runner.run(self.task)

    async def on_first_participant_say_hi(self, transport: DailyTransport, participant):
        if self.daily_params.transcription_enabled:
            transport.capture_participant_transcription(participant["id"])

        language = self._bot_config.llm.language if self._bot_config.llm.language else "en"

        self._bot_config.llm.messages = [LLM_INTRO_PROMPT]
        content = LLM_INTRO_PROMPT["content"] % TO_LLM_LANGUAGE[language]
        self._bot_config.llm.messages[0]["content"] = content

        await self.intro_task.queue_frames(
            [
                images["book1"],
                LLMMessagesFrame(copy.deepcopy(self._bot_config.llm.messages)),
                DailyTransportMessageFrame(CUE_USER_TURN),
                sounds["listening"],
                images["book2"],
                StopTaskFrame(),
            ]
        )

        self._bot_config.llm.messages = [LLM_BASE_PROMPT]
        content = LLM_BASE_PROMPT["content"] % (
            TO_LLM_LANGUAGE[language],
            TO_LLM_LANGUAGE[language],
        )
        self._bot_config.llm.messages[0]["content"] = content
