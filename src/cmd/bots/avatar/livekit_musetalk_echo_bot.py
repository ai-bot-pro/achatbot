import logging

from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from livekit import rtc

from src.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from src.common.types import LivekitParams
from src.transports.livekit import LivekitTransport
from src.cmd.bots.base_livekit import LivekitRoomBot
from src.cmd.bots import register_ai_room_bots
from src.processors.avatar.musetalk_avatar_processor import MusetalkAvatarProcessor
from src.modules.avatar.musetalk import MusetalkAvatar
from src.types.avatar.musetalk import AvatarMuseTalkConfig

from dotenv import load_dotenv

load_dotenv(override=True)


@register_ai_room_bots.register
class LivekitAvatarEchoBot(LivekitRoomBot):
    """
    avatar echo bot
    """

    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()

    def load(self):
        # NOTE: https://github.com/snakers4/silero-vad/discussions/385
        self.vad_analyzer = VADAnalyzerEnvInit.initVADAnalyzerEngine()
        self.livekit_params = LivekitParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_enabled=True,
            audio_out_sample_rate=16000,
            vad_enabled=True,
            vad_analyzer=self.vad_analyzer,
            vad_audio_passthrough=True,
            camera_out_enabled=True,
            camera_out_width=1024,
            camera_out_height=1408,
            camera_out_is_live=True,
        )

        if self._bot_config and self._bot_config.avatar and self._bot_config.avatar.args:
            avatar = MusetalkAvatar(**self._bot_config.avatar.args)
        else:
            avatar = MusetalkAvatar()
        avatar.load()

        config = AvatarMuseTalkConfig(
            input_audio_sample_rate=16000,
            algo_audio_sample_rate=16000,
            output_audio_sample_rate=16000,
            input_audio_slice_duration=1,
            batch_size=avatar.gen_batch_size,
            fps=avatar.fps,
        )

        self.musetalk_processor = MusetalkAvatarProcessor(avatar=avatar, config=config)

    async def arun(self):
        transport = LivekitTransport(
            self.args.token,
            params=self.livekit_params,
        )

        self.task = PipelineTask(
            Pipeline(
                [
                    transport.input_processor(),
                    self.musetalk_processor,
                    transport.output_processor(),
                ]
            ),
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                send_initial_empty_metrics=False,
            ),
        )

        transport.add_event_handler(
            "on_first_participant_joined",
            self.on_first_participant_say_hi,
        )

        logging.info(f"start runing {__name__}")
        await PipelineRunner(handle_sigint=True).run(self.task)

    async def on_first_participant_say_hi(
        self, transport: LivekitTransport, participant: rtc.RemoteParticipant
    ):
        self.session.set_client_id(participant.sid)
