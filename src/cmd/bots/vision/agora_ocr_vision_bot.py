from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.runner import PipelineRunner
from apipeline.pipeline.task import PipelineTask

from src.processors.user_image_request_processor import UserImageTextRequestProcessor
from src.processors.vision.ocr_processor import OCRProcessor
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.cmd.bots.base_agora import AgoraChannelBot
from src.common.types import AgoraParams
from src.transports.agora import AgoraTransport
from .. import register_ai_room_bots


@register_ai_room_bots.register
class AgoraOCRVisionBot(AgoraChannelBot):
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
        agora_params = AgoraParams(
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
        agora_params.audio_out_sample_rate = stream_info["sample_rate"]
        agora_params.audio_out_channels = stream_info["channels"]

        transport = AgoraTransport(self.args.token, params=agora_params)

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(
            transport: AgoraTransport,
            user_id: str,
        ):
            # subscribed the first participant
            transport.capture_participant_video(user_id, framerate=0)

            participant_name = user_id
            image_requester.set_participant_id(user_id)
            await tts_processor.say(
                f"你好,{participant_name}。"
                f"这是一个图像OCR demo。"
                f"对视频中识别的物体请说配置项vision ocr, trigger texts中的识别内容词。"
                f"默认：'识别内容'。"
            )

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
