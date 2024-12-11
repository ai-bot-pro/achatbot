from livekit import rtc
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.runner import PipelineRunner
from apipeline.pipeline.task import PipelineTask

from src.processors.user_image_request_processor import UserImageTextRequestProcessor
from src.processors.vision.ocr_processor import OCRProcessor
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.cmd.bots.base_livekit import LivekitRoomBot
from src.common.types import LivekitParams
from src.transports.livekit import LivekitTransport
from .. import register_ai_room_bots


@register_ai_room_bots.register
class LivekitOCRVisionBot(LivekitRoomBot):
    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()
        self._trigger_texts = (
            self._bot_config.vision_ocr.trigger_texts
            if self._bot_config.vision_ocr.trigger_texts
            else "识别内容。"
        )

    async def arun(self):
        vad_analyzer = self.get_vad_analyzer()
        livekit_params = LivekitParams(
            camera_in_enabled=True,
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=vad_analyzer,
            vad_audio_passthrough=True,
        )

        asr_processor = self.get_asr_processor()
        ocr_processor: OCRProcessor = self.get_vision_ocr_processor()
        tts_processor: TTSProcessor = self.get_tts_processor()
        image_requester = UserImageTextRequestProcessor(
            init_user_prompts=self._trigger_texts,
            desc_img_prompt="",
        )

        stream_info = tts_processor.get_stream_info()
        livekit_params.audio_out_sample_rate = stream_info["sample_rate"]
        livekit_params.audio_out_channels = stream_info["channels"]

        transport = LivekitTransport(self.args.token, params=livekit_params)

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(
            transport: LivekitTransport,
            participant: rtc.RemoteParticipant,
        ):
            # subscribed the first participant
            transport.capture_participant_video(participant.sid, framerate=0)

            participant_name = participant.name if participant.name else participant.identity
            image_requester.set_participant_id(participant.sid)
            await tts_processor.say(
                f"你好,{participant_name}。"
                f"这是一个图像OCR demo。"
                f"对视频中识别的物体请说配置项vision ocr, trigger texts中的识别内容词。"
                f"默认：'识别内容'。"
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
                asr_processor,
                image_requester,
                ocr_processor,
                tts_processor,
                transport.output_processor(),
            ]
        )
        self.task = PipelineTask(pipeline)
        await PipelineRunner().run(self.task)
