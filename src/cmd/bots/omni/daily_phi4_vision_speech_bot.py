import logging

from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.processors.logger import FrameLogger
from apipeline.frames import AudioRawFrame, TextFrame

from src.processors.speech.tts.tts_processor import TTSProcessor
from src.processors.user_image_request_processor import UserImageRequestProcessor
from src.processors.aggregators.vision_image_audio_frame import VisionImageAudioFrameAggregator
from src.processors.speech.audio_save_processor import AudioSaveProcessor
from src.processors.aggregators.user_audio_response import UserAudioResponseAggregator
from src.cmd.bots.base_daily import DailyRoomBot
from src.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from src.common.types import DailyParams
from src.transports.daily import DailyTransport
from src.cmd.bots import register_ai_room_bots
from src.types.frames import *

from dotenv import load_dotenv

load_dotenv(override=True)


@register_ai_room_bots.register
class DailyPhi4VisionSpeechBot(DailyRoomBot):
    """
    use daily images + audio stream(bytes) --> Phi4 vision speech processor -->text/audio_bytes
    """

    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()

    async def arun(self):
        self._vad_analyzer = VADAnalyzerEnvInit.initVADAnalyzerEngine()
        self.params = DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=self._vad_analyzer,
            vad_audio_passthrough=True,
            transcription_enabled=False,
        )

        self._vision_speech_processor = self.get_phi4_vision_speech_processor()
        self._tts_processor: TTSProcessor = self.get_tts_processor()
        stream_info = self._tts_processor.get_stream_info()

        self.params.audio_out_sample_rate = stream_info["sample_rate"]
        self.params.audio_out_channels = stream_info["channels"]

        transport = DailyTransport(
            self.args.room_url,
            self.args.token,
            self.args.bot_name,
            self.params,
        )

        in_audio_aggr = UserAudioResponseAggregator()
        self.image_requester = UserImageRequestProcessor(request_frame_cls=AudioRawFrame)
        image_audio_aggr = VisionImageAudioFrameAggregator()

        # messages = []
        # if self._bot_config.llm.messages:
        #     messages = self._bot_config.llm.messages

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
                    self._vision_speech_processor,
                    FrameLogger(include_frame_types=[TextFrame]),
                    self._tts_processor,
                    FrameLogger(include_frame_types=[AudioRawFrame]),
                    # AudioSaveProcessor(prefix_name="bot_speak"),
                    transport.output_processor(),
                ]
            ),
            params=PipelineParams(
                allow_interruptions=False,
                enable_metrics=False,
                send_initial_empty_metrics=True,
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
        transport.capture_participant_video(participant["id"], framerate=0)
        self.image_requester.set_participant_id(participant["id"])
        await self._tts_processor.say(
            "你好，欢迎使用 Vision Speech Omni Bot. 我是一名虚拟助手，可以结合视频进行提问。"
        )
