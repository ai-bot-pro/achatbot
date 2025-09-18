import logging

from dotenv import load_dotenv
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.processors.logger import FrameLogger
from apipeline.frames import AudioRawFrame, TextFrame

from src.processors.speech.audio_save_processor import AudioSaveProcessor
from src.processors.aggregators.user_audio_response import UserAudioResponseAggregator
from src.cmd.bots.base_daily import DailyRoomBot
from src.common.types import DailyParams
from src.transports.daily import DailyTransport
from src.cmd.bots import register_ai_room_bots
from src.types.frames import PathAudioRawFrame, LLMGenedTokensFrame, BotSpeakingFrame
from .helper import get_step_audio2_processor, get_step_audio2_llm, get_token2wav


load_dotenv(override=True)


@register_ai_room_bots.register
class DailyStepAudio2AQAABot(DailyRoomBot):
    """
    - use daily audio stream(bytes) --> Step2 voice processor --> text/audio_bytes
    """

    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()

        self.vad_analyzer = None
        self.audio_llm = None
        self.token2wav = None

    def load(self):
        self.vad_analyzer = self.get_vad_analyzer()
        self.audio_llm = get_step_audio2_llm(self._bot_config.voice_llm)
        self.token2wav = get_token2wav(self._bot_config.voice_llm)

    async def arun(self):
        assert self.vad_analyzer is not None
        assert self.audio_llm is not None

        self.params = DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=self.vad_analyzer,
            vad_audio_passthrough=True,
        )

        # src/processors/voice/step_audio2_processor.py
        self._voice_processor = get_step_audio2_processor(
            self._bot_config.voice_llm,
            session=self.session,
            audio_llm=self.audio_llm,
            token2wav=self.token2wav,
        )
        if hasattr(self._voice_processor, "stream_info"):
            stream_info = self._voice_processor.stream_info
            self.params.audio_out_sample_rate = stream_info["sample_rate"]
            self.params.audio_out_channels = stream_info["channels"]
        logging.info(f"params: {self.params}")

        transport = DailyTransport(
            self.args.room_url,
            self.args.token,
            self.args.bot_name,
            self.params,
        )

        user_audio_save_processor = None
        bot_speak_audio_save_processor = None
        if self._save_audio:
            user_audio_save_processor = AudioSaveProcessor(
                prefix_name="user_audio_aggr", pass_raw_audio=True
            )
            bot_speak_audio_save_processor = AudioSaveProcessor(
                prefix_name="bot_speak", pass_raw_audio=True
            )

        self.task = PipelineTask(
            Pipeline(
                [
                    transport.input_processor(),
                    UserAudioResponseAggregator(),
                    user_audio_save_processor,
                    FrameLogger(include_frame_types=[AudioRawFrame]),
                    self._voice_processor,
                    FrameLogger(
                        include_frame_types=[TextFrame, AudioRawFrame, LLMGenedTokensFrame]
                    ),
                    bot_speak_audio_save_processor,
                    # FrameLogger(include_frame_types=[BotSpeakingFrame]),
                    transport.output_processor(),  # BotSpeakingFrame
                ]
            ),
            params=PipelineParams(
                allow_interruptions=False,
                enable_metrics=False,
                send_initial_empty_metrics=False,
            ),
        )

        transport.add_event_handlers(
            "on_first_participant_joined",
            [self.on_first_participant_joined, self.on_first_participant_say_hi],
        )
        transport.add_event_handler("on_participant_left", self.on_participant_left)
        transport.add_event_handler("on_call_state_updated", self.on_call_state_updated)

        await PipelineRunner().run(self.task)

    async def on_first_participant_say_hi(self, transport: DailyTransport, participant):
        await self._voice_processor.say(
            "你好，我是一名助手，欢迎语音聊天!",
            temperature=0.7,
            max_new_tokens=1024,
            top_k=20,
            top_p=0.95,
            repetition_penalty=1.1,
        )
