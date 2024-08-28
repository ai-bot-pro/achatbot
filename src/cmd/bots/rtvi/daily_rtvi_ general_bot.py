import argparse
import json
import logging
from typing import Any, Dict, List

from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.pipeline.pipeline import Pipeline
from apipeline.processors.frame_processor import FrameProcessor

from src.processors.aggregators.llm_response import LLMAssistantResponseAggregator, LLMUserResponseAggregator
from src.processors.rtvi_processor import ActionResult, RTVIConfig, RTVIService, RTVIServiceOption, RTVIServiceOptionConfig
from src.processors.llm.base import LLMProcessor
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from src.processors.rtvi_processor import RTVIProcessor
from src.common.types import DailyParams, DailyRoomBotArgs, DailyTranscriptionSettings
from src.transports.daily import DailyTransport
from src.types.ai_conf import AIConfig, LLMConfig
from src.cmd.bots.base import DailyRoomBot
from src.cmd.bots import register_daily_room_bots

from dotenv import load_dotenv
load_dotenv(override=True)


class ASRServiceEventHandler:
    pass


class LLMServiceEventHandler:
    pass


class TTSServiceEventHandler:
    pass


# @register_daily_room_bots.register
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
        self.init_bot_config()
        self.daily_params = DailyParams(
            audio_out_enabled=True,
            transcription_enabled=True,
            vad_enabled=True,
            transcription_settings=DailyTranscriptionSettings(
                language="en",
            ),
        )
        self.runner = PipelineRunner()

    def init_bot_config(self):
        """
        RTVI config options list can transfer to ai bot config
        """
        logging.debug(f'args.bot_config_list: {self.args.bot_config_list}')
        try:
            self.rtvi_config = RTVIConfig(config_list=self.args.bot_config_list)
            self._bot_config: AIConfig = AIConfig(**self.rtvi_config.arguments_dict)
            if self._bot_config.llm is None:
                self._bot_config.llm = LLMConfig()
        except Exception as e:
            raise Exception(f"Failed to parse bot configuration: {e}")
        logging.info(f'ai bot_config: {self._bot_config}')

    async def setup_cb_handler(
            self,
            processor: RTVIProcessor,
            service_name: str,
            option: RTVIServiceOptionConfig):
        logging.debug(f"service_name: {service_name} option: {option}")
        if option.name != "__init__":
            return

        processors: List[FrameProcessor] = []
        try:
            # TODO: maybe list all supported processor
            match service_name:
                case "daily_stream_in":
                    asr_processor = self.get_asr_processor()
                    processors.append(asr_processor)
                case "asr":
                    asr_processor = self.get_asr_processor()
                    processors.append(asr_processor)
                case "llm":
                    llm_processor = self.get_llm_processor()
                    processors.append(llm_processor)
                case "tts":
                    tts_processor = self.get_tts_processor()
                    processors.append(tts_processor)
        except Exception as e:
            logging.warning(f"Exception handle option cb: {e}")

        pipeline = Pipeline(processors=processors)

    async def asr_service_option_change_cb_handler(
            self,
            processor: RTVIProcessor,
            service_name: str,
            option: RTVIServiceOptionConfig):
        logging.debug(f"service_name: {service_name} option: {option}")
        try:
            match option.name:
                case "tag":
                    pass
        except Exception as e:
            logging.warning(f"Exception handle option cb: {e}")

    async def llm_service_option_change_cb_handler(
            self,
            processor: RTVIProcessor,
            service_name: str,
            option: RTVIServiceOptionConfig):
        logging.debug(f"service_name: {service_name} option: {option}")

    async def tts_service_option_change_cb_handler(
            self,
            processor: RTVIProcessor,
            service_name: str,
            option: RTVIServiceOptionConfig):
        logging.debug(f"service_name: {service_name} option: {option}")

    async def service_action_cb_handler(
            self,
            processor: RTVIProcessor,
            service_name: str,
            arguments: Dict[str, Any]) -> ActionResult:
        logging.debug(f"service_name: {service_name} arguments: {arguments}")
        return ActionResult()

    async def arun(self):
        vad_analyzer = VADAnalyzerEnvInit.initVADAnalyzerEngine()
        self.daily_params.vad_analyzer = vad_analyzer

        self.transport = DailyTransport(
            self.args.room_url,
            self.args.token,
            self.args.bot_name,
            self.daily_params,
        )
        rtvi = RTVIProcessor(config=self.rtvi_config)
        rtvi.register_services([
            RTVIService(
                name="asr",
                options=[
                    RTVIServiceOption(
                        name="__init__", type="bool",
                        handler=self.setup_cb_handler,
                    ),
                    RTVIServiceOption(
                        name="tag", type="string",
                        handler=self.asr_service_option_change_cb_handler
                    ),
                ]
            ),
            RTVIService(
                name="llm",
                options=[
                    RTVIServiceOption(
                        name="__init__", type="bool",
                        handler=self.setup_cb_handler,
                    ),
                    RTVIServiceOption(
                        name="tag", type="string",
                        handler=self.llm_service_option_change_cb_handler
                    ),
                ]
            ),
            RTVIService(
                name="tts",
                options=[
                    RTVIServiceOption(
                        name="__init__", type="bool",
                        handler=self.setup_cb_handler,
                    ),
                    RTVIServiceOption(
                        name="tag", type="string",
                        handler=self.tts_service_option_change_cb_handler
                    ),
                ]
            ),
        ])

        pipeline = Pipeline([self.transport.input_processor(), rtvi])

        self.task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                send_initial_empty_metrics=False,
            ))

        self.transport.add_event_handler(
            "on_first_participant_joined",
            self.on_first_participant_joined)
        self.transport.add_event_handler(
            "on_participant_left",
            self.on_participant_left)
        self.transport.add_event_handler(
            "on_call_state_updated",
            self.on_call_state_updated)

        await self.runner.run(self.task)

    async def on_first_participant_joined(self, transport: DailyTransport, participant):
        transport.capture_participant_transcription(participant["id"])
        logging.info("First participant joined")


r"""
python -m src.cmd.bots.daily_rtvi_new_bot -u https://weedge.daily.co/DummyBot   -c $'{"llm":{"model":"llama-3.1-8b-instant","messages":[{"role":"system","content":"You are Chatbot, a friendly, helpful robot. Your output will be converted to audio so don\'t include special characters other than \'\u0021\' or \'?\' in your answers. Respond to what the user said in a creative and helpful way, but keep your responses brief. Start by saying hello."}]},"tts":{"voice":"2ee87190-8f84-4925-97da-e52547f9462c"}}' -t $TOKEN
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTVI Bot Example")
    parser.add_argument("-u", type=str, default="https://weedge.daily.co/chat-bot", help="Room URL")
    parser.add_argument("-t", type=str, default="", help="Token")
    parser.add_argument("-c", type=str, help="Bot configuration blob")
    config = parser.parse_args()

    bot_config = json.loads(config.c) if config.c else {}

    if config.u and config.t and bot_config:
        kwargs = DailyRoomBotArgs(
            bot_config=bot_config,
            room_url=config.u,
            token=config.t,
        ).__dict__
        bot = DailyRTVIGeneralBot(**kwargs)
        bot.run()
    else:
        logging.error("Room URL and Token are required")
