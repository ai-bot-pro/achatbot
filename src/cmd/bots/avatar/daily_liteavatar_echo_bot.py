import logging

from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner

from src.processors.avatar.lite_avatar_processor import LiteAvatarProcessor
from src.modules.avatar.lite_avatar import LiteAvatar
from src.types.avatar.lite_avatar import AvatarInitOption
from src.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from src.common.types import DailyParams
from src.transports.daily import DailyTransport
from src.cmd.bots.base_daily import DailyRoomBot
from src.cmd.bots import register_ai_room_bots
from src.types.frames.data_frames import LLMMessagesFrame

from dotenv import load_dotenv

load_dotenv(override=True)


@register_ai_room_bots.register
class DailyAvatarEchoBot(DailyRoomBot):
    """
    avatar echo bot
    """

    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()
        self.vad_analyzer = None

    def load(self):
        # NOTE: https://github.com/snakers4/silero-vad/discussions/385
        self.vad_analyzer = VADAnalyzerEnvInit.initVADAnalyzerEngine()
        self.lite_avatar_processor = self.get_avatar_processor()

    async def arun(self):
        self.daily_params = DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=self.vad_analyzer,
            vad_audio_passthrough=True,
            camera_out_enabled=True,
            camera_out_width=1024,
            camera_out_height=1408,
            camera_out_is_live=True,
        )

        transport = DailyTransport(
            self.args.room_url,
            self.args.token,
            self.args.bot_name,
            self.daily_params,
        )

        self.task = PipelineTask(
            Pipeline(
                [
                    transport.input_processor(),
                    self.lite_avatar_processor,
                    transport.output_processor(),
                ]
            ),
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                send_initial_empty_metrics=False,
            ),
        )

        transport.add_event_handlers(
            "on_first_participant_joined",
            [self.on_first_participant_joined, self.on_first_participant_say_hi],
        )
        transport.add_event_handler("on_participant_left", self.on_participant_left)
        transport.add_event_handler("on_call_state_updated", self.on_call_state_updated)

        logging.info(f"start runing {__name__}")
        await PipelineRunner(handle_sigint=True).run(self.task)

    async def on_first_participant_say_hi(self, transport: DailyTransport, participant):
        self.session.set_client_id(participant["id"])
