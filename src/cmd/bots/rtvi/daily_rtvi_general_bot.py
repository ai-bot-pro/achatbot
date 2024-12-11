import logging
from typing import Any, Dict, List

from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.pipeline.pipeline import Pipeline
from apipeline.processors.frame_processor import FrameProcessor

from src.processors.aggregators.vision_image_frame import VisionImageFrameAggregator
from src.processors.user_image_request_processor import UserImageRequestProcessor
from src.processors.rtvi.tts_text_processor import RTVITTSTextProcessor
from src.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAIAssistantContextAggregator,
    OpenAIUserContextAggregator,
)
from src.processors.rtvi.rtvi_processor import (
    ActionResult,
    RTVIAction,
    RTVIConfig,
    RTVIService,
    RTVIServiceOption,
    RTVIServiceOptionConfig,
    RTVIProcessor,
)
from src.common.types import DailyParams, DailyTranscriptionSettings
from src.transports.daily import DailyTransport
from src.types.ai_conf import AIConfig, LLMConfig
from src.types.frames.data_frames import LLMMessagesFrame, TextFrame
from src.cmd.bots.base_daily import DailyRoomBot
from src.cmd.bots import register_ai_room_bots


@register_ai_room_bots.register
class DailyRTVIGeneralBot(DailyRoomBot):
    r"""
    use daily (webrtc) transport rtvi general bot
    - init setup pipeline by bot config
    - dynamic config pipeline
    - processor config option change and action event callback handler
    """

    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.rtvi_config: RTVIConfig | None = None
        self._bot_config: AIConfig | None = None
        self._pipeline_params: PipelineParams = PipelineParams()
        self.llm_context = OpenAILLMContext()
        self.daily_params = DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            transcription_enabled=True,
            vad_enabled=True,
            transcription_settings=DailyTranscriptionSettings(
                language="en",
            ),
        )
        self.init_bot_config()
        self.init_processor()

    def init_processor(self):
        # !TODO: services to init
        self.asr_processor = None
        self.image_requester = None
        self.llm_processor = None
        self.tts_processor = None

    def init_bot_config(self):
        """
        RTVI config options list can transfer to ai bot config
        """
        logging.debug(f"args.bot_config_list: {self.args.bot_config_list}")
        try:
            self.rtvi_config = RTVIConfig(config_list=self.args.bot_config_list)
            self._bot_config = AIConfig(**self.rtvi_config._arguments_dict)
            if "pipeline" in self.rtvi_config._arguments_dict:
                self._pipeline_params = PipelineParams(
                    **self.rtvi_config._arguments_dict["pipeline"]
                )
            if "daily_room_stream" in self.rtvi_config._arguments_dict:
                self.daily_params = DailyParams(
                    **self.rtvi_config._arguments_dict["daily_room_stream"]
                )
            if self._bot_config.llm is None:
                self._bot_config.llm = LLMConfig()
        except Exception as e:
            raise Exception(f"Failed to parse bot configuration: {e}")
        logging.info(
            f"daily_params: {self.daily_params},"
            f"pipeline_params: {self._pipeline_params},"
            f"ai bot_config: {self._bot_config}, "
            f"rtvi_config:{self.rtvi_config}"
        )
        if self._bot_config.llm.messages:
            self.llm_context.set_messages(self._bot_config.llm.messages)

    async def vad_service_option_change_cb_handler(
        self, processor: RTVIProcessor, service_name: str, option: RTVIServiceOptionConfig
    ):
        logging.info(f"service_name: {service_name} option: {option}")
        try:
            match option.name:
                case "tag":
                    pass
                case "args":
                    if isinstance(option.value, dict):
                        self.daily_params.vad_analyzer.set_args(**option.value)
        except Exception as e:
            logging.warning(f"Exception handle option cb: {e}")

    async def asr_service_option_change_cb_handler(
        self, processor: RTVIProcessor, service_name: str, option: RTVIServiceOptionConfig
    ):
        logging.info(f"service_name: {service_name} option: {option}")
        try:
            match option.name:
                case "tag":
                    pass
                case "args":
                    if isinstance(option.value, dict):
                        await self.asr_processor.set_asr_args(**option.value)
        except Exception as e:
            logging.warning(f"Exception handle option cb: {e}")

    async def llm_service_option_change_cb_handler(
        self, processor: RTVIProcessor, service_name: str, option: RTVIServiceOptionConfig
    ):
        logging.info(f"service_name: {service_name} option: {option}")
        try:
            match option.name:
                case "tag":
                    pass
                case "args":
                    if isinstance(option.value, dict):
                        self.llm_processor.set_llm_args(**option.value)
                case "model":
                    self.llm_processor.set_model(option.value)
                case "messages":
                    if isinstance(option.value, list):
                        self.llm_context.set_messages(option.value)
        except Exception as e:
            logging.warning(f"Exception handle option cb: {e}")

    async def tts_service_option_change_cb_handler(
        self, processor: RTVIProcessor, service_name: str, option: RTVIServiceOptionConfig
    ):
        logging.info(f"service_name: {service_name} option: {option}")
        try:
            match option.name:
                case "tag":
                    pass
                case "args":
                    if isinstance(option.value, dict):
                        await self.tts_processor.set_tts_args(**option.value)
                        if (
                            "voice_name" in option.value
                            and processor.curr_rtvi_meesage.type == "update-config"
                        ):
                            await self.task.queue_frames(
                                [LLMMessagesFrame(self.llm_context.messages)]
                            )
                case "voice":
                    await self.tts_processor.set_voice(option.value)
                    if processor.curr_rtvi_meesage.type == "update-config":
                        await self.task.queue_frames([LLMMessagesFrame(self.llm_context.messages)])

        except Exception as e:
            logging.warning(f"Exception handle option cb: {e}")

    async def llm_get_ctx_action_cb_handler(
        self, processor: RTVIProcessor, service_name: str, arguments: Dict[str, Any]
    ) -> ActionResult:
        logging.info(f"service_name: {service_name} arguments: {arguments}")

        return self.llm_context.get_messages_json()

    async def llm_set_ctx_action_cb_handler(
        self, processor: RTVIProcessor, service_name: str, arguments: Dict[str, Any]
    ) -> ActionResult:
        logging.info(f"service_name: {service_name} arguments: {arguments}")
        if "interrupt" in arguments and arguments["interrupt"] is True:
            await processor.interrupt_bot()
        if "messages" in arguments:
            self.llm_context.set_messages(arguments["messages"])
        if "tool_choice" in arguments:
            self.llm_context.set_tool_choice(arguments["tool_choice"])
        if "tools" in arguments:
            self.llm_context.set_tools(arguments["tools"])

        return True

    async def llm_append_to_messages_action_cb_handler(
        self, processor: RTVIProcessor, service_name: str, arguments: Dict[str, Any]
    ) -> ActionResult:
        logging.info(f"service_name: {service_name} arguments: {arguments}")
        if "interrupt" in arguments and arguments["interrupt"] is True:
            await processor.interrupt_bot()
        if "messages" in arguments:
            self.llm_context.add_messages(arguments["messages"])
            if "run_immediately" in arguments and arguments["run_immediately"] is True:
                await self.task.queue_frames([LLMMessagesFrame(self.llm_context.messages)])
        return True

    async def llm_run_action_cb_handler(
        self, processor: RTVIProcessor, service_name: str, arguments: Dict[str, Any]
    ) -> ActionResult:
        logging.info(f"service_name: {service_name} arguments: {arguments}")
        if "interrupt" in arguments and arguments["interrupt"] is True:
            await processor.interrupt_bot()
        await self.task.queue_frames([LLMMessagesFrame(self.llm_context.messages)])
        return True

    async def tts_say_action_cb_handler(
        self, processor: RTVIProcessor, service_name: str, arguments: Dict[str, Any]
    ) -> ActionResult:
        logging.info(f"service_name: {service_name} arguments: {arguments}")
        if "interrupt" in arguments and arguments["interrupt"] is True:
            await processor.interrupt_bot()
        if "text" in arguments:
            await self.task.queue_frame(TextFrame(text=arguments["text"]))

        return True

    async def tts_interrupt_action_cb_handler(
        self, processor: RTVIProcessor, service_name: str, arguments: Dict[str, Any]
    ) -> ActionResult:
        logging.info(f"service_name: {service_name} arguments: {arguments}")
        await processor.interrupt_bot()
        return True

    async def arun(self):
        vad_analyzer = self.get_vad_analyzer()
        self.daily_params.vad_analyzer = vad_analyzer

        rtvi = RTVIProcessor(config=self.rtvi_config)
        rtvi.register_services(
            [
                RTVIService(
                    name="vad",
                    options=[
                        RTVIServiceOption(
                            name="tag",
                            type="string",
                            handler=self.vad_service_option_change_cb_handler,
                        ),
                        RTVIServiceOption(
                            name="args",
                            type="dict",
                            handler=self.vad_service_option_change_cb_handler,
                        ),
                    ],
                ),
                RTVIService(
                    name="asr",
                    options=[
                        RTVIServiceOption(
                            name="tag",
                            type="string",
                            handler=self.asr_service_option_change_cb_handler,
                        ),
                        RTVIServiceOption(
                            name="args",
                            type="dict",
                            handler=self.asr_service_option_change_cb_handler,
                        ),
                    ],
                ),
                RTVIService(
                    name="llm",
                    options=[
                        RTVIServiceOption(
                            name="tag",
                            type="string",
                            handler=self.llm_service_option_change_cb_handler,
                        ),
                        RTVIServiceOption(
                            name="model",
                            type="string",
                            handler=self.llm_service_option_change_cb_handler,
                        ),
                        RTVIServiceOption(
                            name="messages",
                            type="list",
                            handler=self.llm_service_option_change_cb_handler,
                        ),
                        RTVIServiceOption(
                            name="args",
                            type="dict",
                            handler=self.llm_service_option_change_cb_handler,
                        ),
                    ],
                ),
                RTVIService(
                    name="tts",
                    options=[
                        RTVIServiceOption(
                            name="tag",
                            type="string",
                            handler=self.tts_service_option_change_cb_handler,
                        ),
                        RTVIServiceOption(
                            name="voice",
                            type="string",
                            handler=self.tts_service_option_change_cb_handler,
                        ),
                        RTVIServiceOption(
                            name="args",
                            type="dict",
                            handler=self.tts_service_option_change_cb_handler,
                        ),
                    ],
                ),
            ]
        )
        rtvi.register_actions(
            [
                RTVIAction(
                    service="llm",
                    action="get_context",
                    handler=self.llm_get_ctx_action_cb_handler,
                ),
                RTVIAction(
                    service="llm",
                    action="set_context",
                    handler=self.llm_set_ctx_action_cb_handler,
                ),
                RTVIAction(
                    service="llm",
                    action="append_to_messages",
                    handler=self.llm_append_to_messages_action_cb_handler,
                ),
                RTVIAction(
                    service="llm",
                    action="run",
                    handler=self.llm_run_action_cb_handler,
                ),
                RTVIAction(
                    service="tts",
                    action="say",
                    handler=self.tts_say_action_cb_handler,
                ),
                RTVIAction(
                    service="tts",
                    action="interrupt",
                    handler=self.tts_interrupt_action_cb_handler,
                ),
            ]
        )

        if self._bot_config.llm:
            llm_user_ctx_aggr = OpenAIUserContextAggregator(self.llm_context)
            llm_assistant_ctx_aggr = OpenAIAssistantContextAggregator(llm_user_ctx_aggr)

        processors: List[FrameProcessor] = []
        try:
            for service_conf in self.rtvi_config.config_list:
                # TODO: maybe list all supported processor
                match service_conf.service:
                    case "asr":
                        self.daily_params.transcription_enabled = False
                        self.daily_params.vad_audio_passthrough = True
                        self.asr_processor = self.get_asr_processor()
                        processors.append(self.asr_processor)
                        processors.append(rtvi)
                    case "llm":
                        if self._bot_config.llm:
                            if self._bot_config.llm.tag and "vision" in self._bot_config.llm.tag:
                                self.image_requester = UserImageRequestProcessor()
                                processors.append(self.image_requester)
                                vision_aggregator = VisionImageFrameAggregator()
                                processors.append(vision_aggregator)
                            else:
                                processors.append(llm_user_ctx_aggr)
                        self.llm_processor = self.get_llm_processor()
                        processors.append(self.llm_processor)
                    case "tts":
                        self.tts_processor = self.get_tts_processor()
                        processors.append(self.tts_processor)
                        tts_text = RTVITTSTextProcessor()
                        processors.append(tts_text)
                        stream_info = self.tts_processor.get_stream_info()
                        self.daily_params.audio_out_sample_rate = stream_info["sample_rate"]
                        self.daily_params.audio_out_channels = stream_info["channels"]
        except Exception as e:
            logging.error(f"init pipeline error: {e}", exc_info=True)
            return

        self.transport = DailyTransport(
            self.args.room_url,
            self.args.token,
            self.args.bot_name,
            self.daily_params,
        )
        if self._bot_config.asr:
            processors = [self.transport.input_processor()] + processors
        else:
            processors = [self.transport.input_processor(), rtvi] + processors
        processors.append(self.transport.output_processor())
        # print(processors)
        if self._bot_config.llm:
            if self._bot_config.llm.tag and "vision" in self._bot_config.llm.tag:
                pass
            else:
                processors.append(llm_assistant_ctx_aggr)
        self.task = PipelineTask(Pipeline(processors), params=self._pipeline_params)

        self.transport.add_event_handler(
            "on_first_participant_joined", self.on_first_participant_joined
        )
        self.transport.add_event_handler("on_participant_left", self.on_participant_left)
        self.transport.add_event_handler("on_call_state_updated", self.on_call_state_updated)

        runner = PipelineRunner()
        await runner.run(self.task)

    async def on_first_participant_joined(self, transport: DailyTransport, participant):
        if self.daily_params.transcription_enabled:
            transport.capture_participant_transcription(participant["id"])
        if (
            self._bot_config.llm
            and self._bot_config.llm.tag
            and "vision" in self._bot_config.llm.tag
        ):
            transport.capture_participant_video(participant["id"], framerate=0)
            self.image_requester and self.image_requester.set_participant_id(participant["id"])

        # joined use tts say "hello" to introduce with llm generate
        if (
            self._bot_config.tts
            and self._bot_config.llm
            and self._bot_config.llm.messages
            and len(self._bot_config.llm.messages) > 0
        ):
            messages = self._bot_config.llm.messages
            messages[0]["content"] = (
                self._bot_config.llm.messages[0]["content"] + " Please introduce yourself first."
            )
            await self.task.queue_frames([LLMMessagesFrame(messages)])
        elif (
            self._bot_config.tts
            and self._bot_config.llm
            and self._bot_config.llm.tag
            and "vision" in self._bot_config.llm.tag
        ):
            hi = "[HI_TEXT] Hello, welcome to use Vision Bot, I am your virtual assistant. u can ask me with video. [/HI_TEXT]"
            match self._bot_config.tts.language:
                case "zh":
                    hi = "[HI_TEXT] 你好，欢迎使用 Vision Bot. 我是一名虚拟助手，可以结合视频进行提问。[/HI_TEXT]"
            await self.task.queue_frame(TextFrame(text=hi))
        logging.info("First participant joined")
