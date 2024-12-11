import logging

from livekit import rtc
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.runner import PipelineRunner
from apipeline.pipeline.task import PipelineTask, PipelineParams
from apipeline.processors.logger import FrameLogger
from apipeline.frames.data_frames import TextFrame

from src.cmd.bots.base_livekit import LivekitRoomBot
from src.processors.vision.vision_processor import MockVisionProcessor
from src.processors.user_image_request_processor import UserImageRequestProcessor
from src.processors.aggregators.user_response import UserResponseAggregator
from src.processors.aggregators.vision_image_frame import VisionImageFrameAggregator
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.common.types import LivekitParams
from src.transports.livekit import LivekitTransport
from src.types.frames.data_frames import UserImageRawFrame
from .. import register_ai_room_bots


@register_ai_room_bots.register
class LivekitMockVisionBot(LivekitRoomBot):
    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()

    async def arun(self):
        vad_analyzer = self.get_vad_analyzer()
        livekit_params = LivekitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=vad_analyzer,
            vad_audio_passthrough=True,
            transcription_enabled=False,
            camera_in_enabled=True,
        )

        asr_processor = self.get_asr_processor()

        tts_processor: TTSProcessor = self.get_tts_processor()
        stream_info = tts_processor.get_stream_info()

        livekit_params.audio_out_sample_rate = stream_info["sample_rate"]
        livekit_params.audio_out_channels = stream_info["channels"]
        transport = LivekitTransport(
            self.args.token,
            params=livekit_params,
        )

        # llm_in_aggr = LLMUserResponseAggregator()
        in_aggr = UserResponseAggregator()
        image_requester = UserImageRequestProcessor()
        vision_aggregator = VisionImageFrameAggregator()
        llm_processor = MockVisionProcessor()
        # llm_out_aggr = LLMAssistantResponseAggregator()

        self.regisiter_room_event(transport)

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(
            transport: LivekitTransport,
            participant: rtc.RemoteParticipant,
        ):
            # subscribed the first participant
            transport.capture_participant_video(participant.sid, framerate=0)
            image_requester.set_participant_id(participant.sid)
            participant_name = participant.name if participant.name else participant.identity
            await transport.send_message(
                f"hello,你好，{participant_name}, 我是机器人。",
                participant_id=participant.identity,
            )
            await tts_processor.say(
                f"你好，{participant_name}, 欢迎使用 Vision Bot. 我是一名虚拟助手，可以结合视频进行提问。"
            )

        @transport.event_handler("on_video_track_subscribed")
        async def on_video_track_subscribed(
            transport: LivekitTransport,
            participant: rtc.RemoteParticipant,
        ):
            transport.capture_participant_video(participant.sid, framerate=0)

        pipeline = Pipeline(
            [
                transport.input_processor(),
                FrameLogger(include_frame_types=[UserImageRawFrame]),
                asr_processor,
                # llm_in_aggr,
                in_aggr,
                image_requester,
                FrameLogger(include_frame_types=[TextFrame]),
                vision_aggregator,
                llm_processor,
                tts_processor,
                transport.output_processor(),
                # llm_out_aggr,
            ]
        )
        self.task = PipelineTask(pipeline, params=PipelineParams())
        await PipelineRunner().run(self.task)
