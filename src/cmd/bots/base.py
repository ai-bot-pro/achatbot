import os
import logging
import asyncio
import uuid

from apipeline.frames.control_frames import EndFrame

from src.common import interface
from src.common.factory import EngineClass
from src.modules.speech.asr import ASREnvInit
from src.modules.speech.tts import TTSEnvInit
from src.processors.speech.asr.asr_processor import ASRProcessor
from src.processors.llm.base import LLMProcessor
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.processors.llm.openai_llm_processor import OpenAILLMProcessor
from src.types.ai_conf import ASRConfig, LLMConfig, TTSConfig, AIConfig
from src.common.types import DailyRoomBotArgs
from src.common.interface import IBot
from src.common.session import Session
from src.common.types import SessionCtx


class DailyRoomBot(IBot):
    r"""
    use ai bot config
    !TODONE: need config processor with bot config (redefine api params) @weedge
    bot config: Dict[str, Dict[str,Any]]
    e.g. {"llm":{"key":val,"tag":TAG,"args":{}}, "tts":{"key":val,"tag":TAG,"args":{}}}
    !TIPS: RTVI config options can transfer to ai bot config
    """

    def __init__(self, **args) -> None:
        self.args = DailyRoomBotArgs(**args)
        if self.args.bot_name is None or len(self.args.bot_name) == 0:
            self.args.bot_name = self.__class__.__name__

        self.task = None
        self.session = Session(**SessionCtx(uuid.uuid4()).__dict__)

        self._bot_config_list = self.args.bot_config_list
        self._bot_config = self.args.bot_config

    def init_bot_config(self):
        try:
            logging.debug(f'args.bot_config: {self.args.bot_config}')
            self._bot_config: AIConfig = AIConfig(**self.args.bot_config)
            if self._bot_config.llm is None:
                self._bot_config.llm = LLMConfig()
        except Exception as e:
            raise Exception(f"Failed to parse bot configuration: {e}")
        logging.info(f'ai bot_config: {self._bot_config}')

    def bot_config(self):
        return self._bot_config

    def run(self):
        try:
            asyncio.run(self.arun())
        except KeyboardInterrupt:
            logging.warning("Ctrl-C detected. Exiting!")

    async def arun(self):
        pass

    async def on_first_participant_joined(self, transport, participant):
        self.session.set_client_id(participant['id'])
        logging.info(f"First participant {participant['id']} joined")

    async def on_participant_left(self, transport, participant, reason):
        if self.task is not None:
            await self.task.queue_frame(EndFrame())
        logging.info("Partcipant left. Exiting.")

    async def on_call_state_updated(self, transport, state):
        logging.info("Call state %s " % state)
        if state == "left" and self.task is not None:
            await self.task.queue_frame(EndFrame())

    def get_asr_processor(self) -> ASRProcessor:
        asr_processor: ASRProcessor = None
        if self._bot_config.asr \
                and self._bot_config.asr.tag == "deepgram_asr_processor" \
                and self._bot_config.asr.args:
            from src.processors.speech.asr.deepgram_asr_processor import DeepgramAsrProcessor
            asr_processor = DeepgramAsrProcessor(
                api_key=os.getenv("DEEPGRAM_API_KEY"),
                **self._bot_config.asr.args)
        else:
            # use asr engine processor
            asr: interface.IAsr | EngineClass | None = None
            if self._bot_config.asr \
                    and self._bot_config.asr.tag \
                    and self._bot_config.asr.args:
                asr = ASREnvInit.getEngine(
                    self._bot_config.asr.tag, **self._bot_config.asr.args)
            else:
                logging.info(f"use default asr engine processor")
                asr = ASREnvInit.initASREngine()
                self._bot_config.asr = ASRConfig(tag=asr.SELECTED_TAG, args=asr.get_args_dict())
            asr_processor = ASRProcessor(
                asr=asr,
                session=self.session
            )
        return asr_processor

    def get_llm_processor(self) -> LLMProcessor:
        # default use openai llm processor
        api_key = os.environ.get("OPENAI_API_KEY")
        if "groq" in self._bot_config.llm.base_url:
            api_key = os.environ.get("GROQ_API_KEY")
        elif "together" in self._bot_config.llm.base_url:
            api_key = os.environ.get("TOGETHER_API_KEY")
        llm_processor = OpenAILLMProcessor(
            model=self._bot_config.llm.model,
            base_url=self._bot_config.llm.base_url,
            api_key=api_key,
        )
        return llm_processor

    def get_tts_processor(self) -> TTSProcessor:
        tts_processor: TTSProcessor | None = None
        if self._bot_config.tts and self._bot_config.tts.tag == "elevenlabs_tts_processor":
            from src.processors.speech.tts.elevenlabs_tts_processor import ElevenLabsTTSProcessor
            tts_processor = ElevenLabsTTSProcessor(**self._bot_config.tts.args)
        elif self._bot_config.tts and self._bot_config.tts.tag == "cartesia_tts_processor":
            from src.processors.speech.tts.cartesia_tts_processor import CartesiaTTSProcessor
            tts_processor = CartesiaTTSProcessor(
                # voice_id=self._bot_config.tts.voice,
                # cartesia_version="2024-06-10",
                # model_id="sonic-multilingual",
                # language=self._bot_config.tts.language if self._bot_config.tts.language else "en",
                **self._bot_config.tts.args
            )
        else:
            # use tts engine processor
            tts: interface.ITts | EngineClass | None = None
            if self._bot_config.tts \
                    and self._bot_config.tts.tag \
                    and self._bot_config.tts.args:
                tts = TTSEnvInit.getEngine(
                    self._bot_config.tts.tag, **self._bot_config.tts.args)
            else:
                # default tts engine processor
                logging.info(f"use default tts engine processor")
                tts = TTSEnvInit.initTTSEngine()
                self._bot_config.tts = TTSConfig(tag=tts.SELECTED_TAG, args=tts.get_args_dict())

            tts_processor = TTSProcessor(tts=tts, session=self.session)

        return tts_processor
