import logging

from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.processors.logger import FrameLogger
from apipeline.frames import AudioRawFrame, TextFrame

from src.processors.user_image_request_processor import UserImageRequestProcessor
from src.processors.aggregators.vision_image_audio_frame import VisionImageAudioFrameAggregator
from src.processors.speech.audio_save_processor import AudioSaveProcessor
from src.processors.aggregators.user_audio_response import UserAudioResponseAggregator
from src.cmd.bots.base_livekit import LivekitRoomBot, rtc
from src.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from src.common.types import LivekitParams
from src.transports.livekit import LivekitTransport
from src.cmd.bots import register_ai_room_bots
from src.types.frames import *

from dotenv import load_dotenv

load_dotenv(override=True)


@register_ai_room_bots.register
class LivekitQwen2_5OmniVisionVoiceBot(LivekitRoomBot):
    """
    use livekit images + audio stream(bytes) --> Qwen2_5Omni vision voice processor -->text/audio_bytes
    """

    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()

    async def arun(self):
        self._vad_analyzer = VADAnalyzerEnvInit.initVADAnalyzerEngine()
        self.params = LivekitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=self._vad_analyzer,
            vad_audio_passthrough=True,
            camera_in_enabled=True,
        )

        self._vision_voice_processor = self.get_qwen2_5omni_vision_voice_processor()
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

        self.task = PipelineTask(
            Pipeline(
                [
                    transport.input_processor(),
                    in_audio_aggr,
                    FrameLogger(include_frame_types=[AudioRawFrame]),
                    # AudioSaveProcessor(prefix_name="user_audio_aggr"),
                    # FrameLogger(include_frame_types=[PathAudioRawFrame]),
                    self.image_requester,
                    image_audio_aggr,
                    FrameLogger(include_frame_types=[VisionImageVoiceRawFrame]),
                    self._vision_voice_processor,
                    FrameLogger(include_frame_types=[AudioRawFrame, TextFrame]),
                    # AudioSaveProcessor(prefix_name="bot_speak"),
                    transport.output_processor(),
                ]
            ),
            params=PipelineParams(
                allow_interruptions=False,
                enable_metrics=True,
                send_initial_empty_metrics=False,
            ),
        )

        await PipelineRunner().run(self.task)

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
