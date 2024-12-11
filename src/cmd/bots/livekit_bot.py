import logging

from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner

from src.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator,
)
from src.processors.llm.base import LLMProcessor
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.common.types import LivekitParams
from src.transports.livekit import LivekitTransport
from src.cmd.bots.base_livekit import LivekitRoomBot
from src.cmd.bots import register_ai_room_bots
from src.types.frames.data_frames import LLMMessagesFrame

from dotenv import load_dotenv

load_dotenv(override=True)


@register_ai_room_bots.register
class LivekitBot(LivekitRoomBot):
    """
    audio chat with livekit webRTC room bot
    """

    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()

    async def arun(self):
        vad_analyzer = self.get_vad_analyzer()
        params = LivekitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=vad_analyzer,
            vad_audio_passthrough=True,
        )

        asr_processor = self.get_asr_processor()
        llm_processor: LLMProcessor = self.get_llm_processor()
        tts_processor: TTSProcessor = self.get_tts_processor()
        stream_info = tts_processor.get_stream_info()
        params.audio_out_sample_rate = stream_info["sample_rate"]
        params.audio_out_channels = stream_info["channels"]

        transport = LivekitTransport(
            self.args.token,
            params=params,
        )

        messages = []
        if self._bot_config.llm.messages:
            messages = self._bot_config.llm.messages

        user_response = LLMUserResponseAggregator(messages)
        assistant_response = LLMAssistantResponseAggregator(messages)

        self.task = PipelineTask(
            Pipeline(
                [
                    transport.input_processor(),
                    asr_processor,
                    user_response,
                    llm_processor,
                    tts_processor,
                    transport.output_processor(),
                    assistant_response,
                ]
            ),
            params=PipelineParams(
                # TODO: open interruptions some issue when sub remote participant
                allow_interruptions=False,
                enable_metrics=True,
                send_initial_empty_metrics=False,
            ),
        )

        transport.add_event_handlers(
            "on_first_participant_joined", [self.on_first_participant_say_hi]
        )

        await PipelineRunner().run(self.task)

    async def on_first_participant_say_hi(self, transport: LivekitTransport, participant):
        # joined use tts say "hello" to introduce with llm generate
        if (
            self._bot_config.tts
            and self._bot_config.llm
            and self._bot_config.llm.messages
            and len(self._bot_config.llm.messages) == 1
        ):
            hi_text = "Please introduce yourself first."
            if self._bot_config.llm.language and self._bot_config.llm.language == "zh":
                hi_text = "请用中文介绍下自己。"
            self._bot_config.llm.messages.append(
                {
                    "role": "user",
                    "content": hi_text,
                }
            )
            await self.task.queue_frames([LLMMessagesFrame(self._bot_config.llm.messages)])
