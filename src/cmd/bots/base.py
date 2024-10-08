import os
import logging
import asyncio
import uuid

from apipeline.frames.control_frames import EndFrame
from apipeline.pipeline.task import PipelineTask

from src.modules.vision.ocr import VisionOCREnvInit
from src.modules.vision.detector import VisionDetectorEnvInit
from src.processors.ai_processor import AIProcessor
from src.processors.vision.vision_processor import MockVisionProcessor
from src.processors.speech.asr.base import ASRProcessorBase
from src.processors.llm.base import LLMProcessor
from src.processors.speech.tts.base import TTSProcessorBase
from src.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from src.modules.speech.asr import ASREnvInit
from src.core.llm import LLMEnvInit
from src.modules.speech.tts import TTSEnvInit
from src.types.ai_conf import ASRConfig, LLMConfig, TTSConfig, AIConfig
from src.common import interface
from src.common.factory import EngineClass
from src.common.types import RoomBotArgs
from src.common.interface import IBot, IVisionDetector
from src.common.session import Session
from src.common.types import SessionCtx

from dotenv import load_dotenv

load_dotenv(override=True)


class AIRoomBot(IBot):
    r"""
    use ai bot config
    !TODONE: need config processor with bot config (redefine api params) @weedge
    bot config: Dict[str, Dict[str,Any]]
    e.g. {"llm":{"key":val,"tag":TAG,"args":{}}, "tts":{"key":val,"tag":TAG,"args":{}}}
    !TIPS: RTVI config options can transfer to ai bot config
    """

    def __init__(self, **args) -> None:
        self.args = RoomBotArgs(**args)
        if self.args.bot_name is None or len(self.args.bot_name) == 0:
            self.args.bot_name = self.__class__.__name__

        self.task: PipelineTask = None
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
        asyncio.run(self.try_run())

    async def try_run(self):
        try:
            await self.arun()
        except asyncio.CancelledError:
            logging.info("CancelledError, Exiting!")
            if self.task is not None:
                await self.task.queue_frame(EndFrame())
        except KeyboardInterrupt:
            logging.warning("Ctrl-C detected. Exiting!")
            if self.task is not None:
                await self.task.queue_frame(EndFrame())
        except Exception as e:
            logging.error(f"run error: {e}", exc_info=True)

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

    def get_vad_analyzer(self) -> interface.IVADAnalyzer | EngineClass:
        vad_analyzer: interface.IVADAnalyzer | EngineClass = None
        if self._bot_config.vad and self._bot_config.vad.tag \
                and len(self._bot_config.vad.tag) > 0  \
                and self._bot_config.vad.args:
            vad_analyzer = VADAnalyzerEnvInit.getEngine(
                self._bot_config.vad.tag, **self._bot_config.vad.args)
        else:
            vad_analyzer = VADAnalyzerEnvInit.initVADAnalyzerEngine()
        return vad_analyzer

    def get_asr_processor(self) -> ASRProcessorBase:
        asr_processor: ASRProcessorBase | None = None
        if self._bot_config.asr and self._bot_config.asr.tag \
                and self._bot_config.asr.tag == "deepgram_asr_processor" \
                and self._bot_config.asr.args:
            from src.processors.speech.asr.deepgram_asr_processor import DeepgramAsrProcessor
            asr_processor = DeepgramAsrProcessor(
                api_key=os.getenv("DEEPGRAM_API_KEY"),
                **self._bot_config.asr.args)
        else:
            # use asr engine processor
            from src.processors.speech.asr.asr_processor import ASRProcessor
            asr: interface.IAsr | EngineClass | None = None
            if self._bot_config.asr and self._bot_config.asr.tag \
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

    def get_vision_llm_processor(self) -> LLMProcessor:
        from src.processors.vision.vision_processor import VisionProcessor
        llm_config = self._bot_config.llm
        if self._bot_config.vision_llm:
            llm_config = self._bot_config.vision_llm
        if "mock" in llm_config.tag:
            llm_processor = MockVisionProcessor()
        else:
            logging.debug(f"init engine llm processor tag: {llm_config.tag}")
            llm = LLMEnvInit.initLLMEngine(llm_config.tag, llm_config.args)
            llm_processor = VisionProcessor(llm, self.session)
        return llm_processor

    def get_vision_annotate_processor(self) -> AIProcessor:
        from src.processors.vision.annotate_processor import AnnotateProcessor
        detector: IVisionDetector | EngineClass = VisionDetectorEnvInit.initVisionDetectorEngine(
            self._bot_config.vision_detector.tag,
            self._bot_config.vision_detector.args)
        processor = AnnotateProcessor(detector, self.session)
        return processor

    def get_vision_detect_processor(self) -> AIProcessor:
        from src.processors.vision.detect_processor import DetectProcessor
        detector = VisionDetectorEnvInit.initVisionDetectorEngine(
            self._bot_config.vision_detector.tag,
            self._bot_config.vision_detector.args)

        desc, out_desc = "", ""
        if "desc" in self._bot_config.vision_detector.args:
            desc = self._bot_config.vision_detector.args["desc"]
        if "out_desc" in self._bot_config.vision_detector.args:
            out_desc = self._bot_config.vision_detector.args["out_desc"]
        if desc and out_desc:
            processor = DetectProcessor(detected_text=desc,
                                        out_detected_text=out_desc,
                                        detector=detector, session=self.session)
        elif desc:
            processor = DetectProcessor(detected_text=desc,
                                        detector=detector, session=self.session)
        elif out_desc:
            processor = DetectProcessor(out_detected_text=out_desc,
                                        detector=detector, session=self.session)
        else:
            processor = DetectProcessor(detector=detector, session=self.session)

        return processor

    def get_vision_ocr_processor(self) -> AIProcessor:
        from src.processors.vision.ocr_processor import OCRProcessor
        ocr = VisionOCREnvInit.initVisionOCREngine(
            self._bot_config.vision_ocr.tag,
            self._bot_config.vision_ocr.args)
        processor = OCRProcessor(ocr=ocr, session=self.session)
        return processor

    def get_openai_llm_processor(self) -> LLMProcessor:
        from src.processors.llm.openai_llm_processor import OpenAILLMProcessor, OpenAIGroqLLMProcessor
        # default use openai llm processor
        api_key = os.environ.get("OPENAI_API_KEY")
        if "groq" in self._bot_config.llm.base_url:
            # https://console.groq.com/docs/models
            api_key = os.environ.get("GROQ_API_KEY")
            llm_processor = OpenAIGroqLLMProcessor(
                model=self._bot_config.llm.model,
                base_url=self._bot_config.llm.base_url,
                api_key=api_key,
            )
            return llm_processor
        elif "together" in self._bot_config.llm.base_url:
            # https://docs.together.ai/docs/chat-models
            api_key = os.environ.get("TOGETHER_API_KEY")
        llm_processor = OpenAILLMProcessor(
            model=self._bot_config.llm.model,
            base_url=self._bot_config.llm.base_url,
            api_key=api_key,
        )
        return llm_processor

    def get_llm_processor(self) -> LLMProcessor:
        if self._bot_config.llm and self._bot_config.llm.tag \
                and "vision" in self._bot_config.llm.tag:
            # engine llm processor(just support vision model, other TODO):
            # (llm_llamacpp, llm_personalai_proxy, llm_transformers etc..)
            llm_processor = self.get_vision_llm_processor()
        else:
            llm_processor = self.get_openai_llm_processor()
        return llm_processor

    def get_tts_processor(self) -> TTSProcessorBase:
        tts_processor: TTSProcessorBase | None = None
        if self._bot_config.tts and self._bot_config.tts.tag and self._bot_config.tts.args:
            if self._bot_config.tts.tag == "elevenlabs_tts_processor":
                from src.processors.speech.tts.elevenlabs_tts_processor import ElevenLabsTTSProcessor
                tts_processor = ElevenLabsTTSProcessor(**self._bot_config.tts.args)
            elif self._bot_config.tts.tag == "cartesia_tts_processor":
                from src.processors.speech.tts.cartesia_tts_processor import CartesiaTTSProcessor
                tts_processor = CartesiaTTSProcessor(**self._bot_config.tts.args)
            else:
                # use tts engine processor
                from src.processors.speech.tts.tts_processor import TTSProcessor
                tts = TTSEnvInit.getEngine(
                    self._bot_config.tts.tag, **self._bot_config.tts.args)
                self._bot_config.tts.tag = tts.SELECTED_TAG,
                self._bot_config.tts.args = tts.get_args_dict()
                tts_processor = TTSProcessor(tts=tts, session=self.session)
        else:
            # default tts engine processor
            from src.processors.speech.tts.tts_processor import TTSProcessor
            logging.info(f"use default tts engine processor")
            tag = None
            if self._bot_config.tts and self._bot_config.tts.tag:
                tag = self._bot_config.tts.tag
            tts = TTSEnvInit.initTTSEngine(tag)
            self._bot_config.tts = TTSConfig(tag=tts.SELECTED_TAG, args=tts.get_args_dict())
            tts_processor = TTSProcessor(tts=tts, session=self.session)

        return tts_processor
