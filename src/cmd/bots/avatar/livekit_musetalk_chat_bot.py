import logging

from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from livekit import rtc

from src.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator,
)
from src.processors.llm.base import LLMProcessor
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from src.common.types import LivekitParams
from src.transports.livekit import LivekitTransport
from src.cmd.bots.base_livekit import LivekitRoomBot
from src.cmd.bots import register_ai_room_bots
from src.types.frames.data_frames import LLMMessagesFrame
from src.processors.avatar.musetalk_avatar_processor import MusetalkAvatarProcessor
from src.modules.avatar.musetalk import MusetalkAvatar
from src.types.avatar.musetalk import AvatarMuseTalkConfig

from dotenv import load_dotenv

load_dotenv(override=True)


@register_ai_room_bots.register
class LivekitAvatarChatBot(LivekitRoomBot):
    """
    avatar chat bot
    """

    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()

    def load(self):
        # NOTE: https://github.com/snakers4/silero-vad/discussions/385
        self.vad_analyzer = VADAnalyzerEnvInit.initVADAnalyzerEngine()

        if self._bot_config and self._bot_config.avatar and self._bot_config.avatar.args:
            self.avatar = MusetalkAvatar(**self._bot_config.avatar.args)
        else:
            self.avatar = MusetalkAvatar()
        self.avatar.load()

    async def arun(self):
        self.livekit_params = LivekitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=self.vad_analyzer,
            vad_audio_passthrough=True,
            transcription_enabled=False,
            camera_out_enabled=True,
            camera_out_width=1024,
            camera_out_height=1408,
            camera_out_is_live=True,
        )

        asr_processor = self.get_asr_processor()

        llm_processor: LLMProcessor = self.get_llm_processor()

        tts_processor: TTSProcessor = self.get_tts_processor()
        stream_info = tts_processor.get_stream_info()
        self.livekit_params.audio_out_sample_rate = stream_info["sample_rate"]
        self.livekit_params.audio_out_channels = stream_info["channels"]

        config = AvatarMuseTalkConfig(
            input_audio_sample_rate=stream_info["sample_rate"],
            algo_audio_sample_rate=16000,
            output_audio_sample_rate=16000,
            input_audio_slice_duration=1,
            batch_size=self.avatar.gen_batch_size,
            fps=self.avatar.fps,
        )
        musetalk_processor = MusetalkAvatarProcessor(avatar=self.avatar, config=config)

        transport = LivekitTransport(
            self.args.token,
            params=self.livekit_params,
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
                    musetalk_processor,
                    transport.output_processor(),
                    assistant_response,
                ]
            ),
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                send_initial_empty_metrics=False,
            ),
        )

        self.regisiter_room_event(transport)
        transport.add_event_handler(
            "on_first_participant_joined",
            self.on_first_participant_say_hi,
        )

        await PipelineRunner().run(self.task)

    async def on_first_participant_say_hi(
        self, transport: LivekitTransport, participant: rtc.RemoteParticipant
    ):
        self.session.set_client_id(participant.sid)
        name = participant.name or participant.identity or "weedge"

        # joined use tts say "hello" to introduce with llm generate
        if (
            self._bot_config.tts
            and self._bot_config.llm
            and self._bot_config.llm.messages is not None
            and isinstance(self._bot_config.llm.messages, list)
        ):
            hi_text = "Please introduce yourself first."
            if self._bot_config.llm.language and self._bot_config.llm.language == "zh":
                hi_text = f"你好，我叫{name}, 请用中文介绍下自己。"
            self._bot_config.llm.messages.append(
                {
                    "role": "user",
                    "content": hi_text,
                }
            )
            await self.task.queue_frames([LLMMessagesFrame(self._bot_config.llm.messages)])
