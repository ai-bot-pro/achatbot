import logging

from dotenv import load_dotenv
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.processors.logger import FrameLogger
from apipeline.frames import AudioRawFrame, TextFrame

from src.cmd.bots.omni.helper import get_qwen3omni_llm, get_qwen3omni_processor
from src.processors.user_image_request_processor import UserImageRequestProcessor
from src.processors.aggregators.vision_image_audio_frame import VisionImageAudioFrameAggregator
from src.processors.speech.audio_save_processor import AudioSaveProcessor
from src.processors.aggregators.user_audio_response import UserAudioResponseAggregator
from src.cmd.bots.base_livekit import LivekitRoomBot, rtc
from src.common.types import LivekitParams
from src.transports.livekit import LivekitTransport
from src.cmd.bots import register_ai_room_bots
from src.types.frames import *


load_dotenv(override=True)


@register_ai_room_bots.register
class LivekitQwen3OmniVisionVoiceBot(LivekitRoomBot):
    """
    use livekit images + audio stream(bytes) --> Qwen3Omni vision voice processor -->text/audio_bytes
    """

    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()
        self.llm_config = self._bot_config.omni_llm or self._bot_config.llm

        self.vad_analyzer = None
        self.omni_llm = None

    def load(self):
        self.vad_analyzer = self.get_vad_analyzer()
        self.omni_llm = get_qwen3omni_llm(self.llm_config)

    async def arun(self):
        self.params = LivekitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=self.vad_analyzer,
            vad_audio_passthrough=True,
            camera_in_enabled=True,
        )

        self._vision_voice_processor = get_qwen3omni_processor(
            llm_config=self.llm_config,
            session=self.session,
            llm=self.omni_llm,
        )
        stream_info = self._vision_voice_processor.stream_info
        self.params.audio_out_sample_rate = stream_info["sample_rate"]
        self.params.audio_out_channels = stream_info["channels"]

        transport = LivekitTransport(
            self.args.token,
            params=self.params,
        )
        self.regisiter_room_event(transport)

        in_audio_aggr = UserAudioResponseAggregator()
        self.image_requester = UserImageRequestProcessor(request_frame_cls=AudioRawFrame)
        image_audio_aggr = VisionImageAudioFrameAggregator()

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
                    in_audio_aggr,
                    user_audio_save_processor,
                    FrameLogger(include_frame_types=[AudioRawFrame]),
                    self.image_requester,
                    image_audio_aggr,
                    FrameLogger(include_frame_types=[VisionImageVoiceRawFrame]),
                    self._vision_voice_processor,
                    FrameLogger(include_frame_types=[AudioRawFrame, TextFrame]),
                    bot_speak_audio_save_processor,
                    transport.output_processor(),
                ]
            ),
            params=PipelineParams(
                allow_interruptions=False,
                enable_metrics=True,
                send_initial_empty_metrics=False,
            ),
        )

        await PipelineRunner(handle_sigint=self._handle_sigint).run(self.task)

    async def on_first_participant_joined(
        self,
        transport: LivekitTransport,
        participant: rtc.RemoteParticipant,
    ):
        # subscribed the first participant
        transport.capture_participant_video(participant.sid, framerate=0)
        self.image_requester.set_participant_id(participant.sid)

        participant_name = participant.name if participant.name else participant.identity
        await self._vision_voice_processor.say(
            f"你好，{participant_name} 欢迎使用 Vision Voice Omni Bot. 我是一名虚拟助手，可以结合视频进行提问。"
        )

    async def on_video_track_subscribed(
        self,
        transport: LivekitTransport,
        participant: rtc.RemoteParticipant,
    ):
        transport.capture_participant_video(participant.sid, framerate=0)
