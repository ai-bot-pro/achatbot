from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.runner import PipelineRunner
from apipeline.pipeline.task import PipelineTask

from src.processors.user_image_request_processor import UserImageTextRequestProcessor
from src.processors.vision.ocr_processor import OCRProcessor
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.common.types import DailyParams
from src.cmd.bots.base_daily import DailyRoomBot
from src.transports.daily import DailyTransport
from .. import register_ai_room_bots


@register_ai_room_bots.register
class DailyOCRVisionBot(DailyRoomBot):
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
        daily_params = DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=vad_analyzer,
            vad_audio_passthrough=True,
            transcription_enabled=False,
        )

        self.asr_processor = self.get_asr_processor()
        self.ocr_processor: OCRProcessor = self.get_vision_ocr_processor()
        self.tts_processor: TTSProcessor = self.get_tts_processor()

        stream_info = self.tts_processor.get_stream_info()
        daily_params.audio_out_sample_rate = stream_info["sample_rate"]
        daily_params.audio_out_channels = stream_info["channels"]

        transport = DailyTransport(
            self.args.room_url, self.args.token, self.args.bot_name, daily_params
        )
        transport.add_event_handler("on_first_participant_joined", self.on_first_participant_joined)
        transport.add_event_handler("on_participant_left", self.on_participant_left)
        transport.add_event_handler("on_call_state_updated", self.on_call_state_updated)

        self.image_requester = UserImageTextRequestProcessor(
            init_user_prompts=self._trigger_texts,
            desc_img_prompt="",
        )

        pipeline = Pipeline(
            [
                transport.input_processor(),
                self.asr_processor,
                self.image_requester,
                self.ocr_processor,
                self.tts_processor,
                transport.output_processor(),
            ]
        )
        self.task = PipelineTask(pipeline)
        await PipelineRunner().run(self.task)

    async def on_first_participant_joined(self, transport: DailyTransport, participant):
        transport.capture_participant_video(participant["id"], framerate=0)
        self.image_requester.set_participant_id(participant["id"])
        await self.tts_processor.say(
            "你好。这是一个图像OCR demo。"
            "对视频中识别的物体请说配置项vision ocr, trigger texts中的识别内容词。"
            "默认：'识别内容'。"
        )
