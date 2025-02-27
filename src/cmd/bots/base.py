import os
import logging
import asyncio
import sys
import uuid

from apipeline.frames.control_frames import EndFrame
from apipeline.pipeline.task import PipelineTask

from src.processors.omni.base import VisionVoiceProcessorBase
from src.processors.voice.base import VoiceProcessorBase
from src.processors.image.base import ImageGenProcessor
from src.processors.image import get_image_gen_processor
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
from src.types.ai_conf import (
    TOGETHER_LLM_MODEL,
    TOGETHER_LLM_URL,
    ASRConfig,
    LLMConfig,
    TTSConfig,
    AIConfig,
)
from src.common import interface
from src.common.factory import EngineClass
from src.common.types import BotRunArgs
from src.common.interface import IBot, IVisionDetector
from src.common.session import Session
from src.common.types import SessionCtx

from dotenv import load_dotenv

load_dotenv(override=True)


class AIBot(IBot):
    r"""
    use ai bot config
    !TODONE: need config processor with bot config (redefine api params) @weedge
    bot config: Dict[str, Dict[str,Any]]
    e.g. {"llm":{"key":val,"tag":TAG,"args":{}}, "tts":{"key":val,"tag":TAG,"args":{}}}
    !TIPS: RTVI config options can transfer to ai bot config

    !NOTE:
    use multiprocessing pipe to run bot with unix socket, bot __init__ to new a bot obj must be serializable (pickle); or wraper a func, don't use bot obj methed.
    """

    def __init__(self, **args) -> None:
        self.args = BotRunArgs(**args)
        if self.args.bot_name is None or len(self.args.bot_name) == 0:
            self.args.bot_name = self.__class__.__name__

        self.task: PipelineTask | None = None
        self.session = Session(**SessionCtx(uuid.uuid4()).__dict__)

        self._bot_config_list = self.args.bot_config_list
        self._bot_config = self.args.bot_config
        self._handle_sigint = self.args.handle_sigint

    def init_bot_config(self):
        try:
            logging.debug(f"args.bot_config: {self.args.bot_config}")
            self._bot_config: AIConfig = AIConfig(**self.args.bot_config)
            if self._bot_config.llm is None:
                self._bot_config.llm = LLMConfig()
        except Exception as e:
            raise Exception(f"Failed to parse bot configuration: {e}")
        logging.info(f"ai bot_config: {self._bot_config}")

    def bot_config(self):
        return self._bot_config

    def load(self):
        # !TODO: load model ckpt when bot start @weedge
        # when deploy need load model ckpt, then run serve
        # now just support one person with one bot agent
        pass

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

    def get_vad_analyzer(self) -> interface.IVADAnalyzer | EngineClass:
        vad_analyzer: interface.IVADAnalyzer | EngineClass = None
        if (
            self._bot_config.vad
            and self._bot_config.vad.tag
            and len(self._bot_config.vad.tag) > 0
            and self._bot_config.vad.args
        ):
            vad_analyzer = VADAnalyzerEnvInit.getEngine(
                self._bot_config.vad.tag, **self._bot_config.vad.args
            )
        else:
            vad_analyzer = VADAnalyzerEnvInit.initVADAnalyzerEngine()
        return vad_analyzer

    def get_asr_processor(self) -> ASRProcessorBase:
        asr_processor: ASRProcessorBase | None = None
        if (
            self._bot_config.asr
            and self._bot_config.asr.tag
            and self._bot_config.asr.tag == "deepgram_asr_processor"
            and self._bot_config.asr.args
        ):
            from src.processors.speech.asr.deepgram_asr_processor import DeepgramAsrProcessor

            asr_processor = DeepgramAsrProcessor(
                api_key=os.getenv("DEEPGRAM_API_KEY"), **self._bot_config.asr.args
            )
        else:
            # use asr engine processor
            from src.processors.speech.asr.asr_processor import ASRProcessor

            asr: interface.IAsr | EngineClass | None = None
            if self._bot_config.asr and self._bot_config.asr.tag and self._bot_config.asr.args:
                asr = ASREnvInit.getEngine(self._bot_config.asr.tag, **self._bot_config.asr.args)
            else:
                logging.info("use default asr engine processor")
                asr = ASREnvInit.initASREngine()
                self._bot_config.asr = ASRConfig(tag=asr.SELECTED_TAG, args=asr.get_args_dict())
            asr_processor = ASRProcessor(asr=asr, session=self.session)
        return asr_processor

    def get_vision_llm_processor(self, llm_config: LLMConfig | None = None) -> LLMProcessor:
        """
        get local vision llm
        """
        from src.processors.vision.vision_processor import VisionProcessor

        if not llm_config:
            llm_config = self._bot_config.vision_llm
        if "mock" in llm_config.tag:
            llm_processor = MockVisionProcessor()
        else:
            logging.debug(f"init engine llm processor tag: {llm_config.tag}")
            llm_engine = LLMEnvInit.initLLMEngine(llm_config.tag, llm_config.args)
            llm_processor = VisionProcessor(llm_engine, self.session)
        return llm_processor

    def get_remote_llm_processor(self, llm: LLMConfig | None = None) -> LLMProcessor:
        """
        get remote llm
        """
        if not llm:
            llm = self._bot_config.llm
        if llm and llm.tag and "google" in llm.tag:
            llm_processor = self.get_google_llm_processor(llm)
        elif llm and llm.tag and "litellm" in llm.tag:
            llm_processor = self.get_litellm_processor(llm)
        elif llm and llm.tag and "openai" in llm.tag:
            llm_processor = self.get_openai_llm_processor(llm)
        else:
            llm_processor = self.get_google_llm_processor(llm)
        return llm_processor

    def get_openai_llm_processor(self, llm: LLMConfig | None = None) -> LLMProcessor:
        from src.processors.llm.openai_llm_processor import (
            OpenAILLMProcessor,
            OpenAIGroqLLMProcessor,
        )

        if not llm:
            llm = self._bot_config.llm
        # default use openai llm processor
        api_key = os.environ.get("OPENAI_API_KEY")
        if llm:
            if "groq" in llm.base_url:
                # https://console.groq.com/docs/models
                api_key = os.environ.get("GROQ_API_KEY")
                llm_processor = OpenAIGroqLLMProcessor(
                    model=llm.model,
                    base_url=llm.base_url,
                    api_key=api_key,
                )
                return llm_processor
            elif "together" in llm.base_url:
                # https://docs.together.ai/docs/chat-models
                api_key = os.environ.get("TOGETHER_API_KEY")
            else:
                llm.base_url = TOGETHER_LLM_URL
                llm.model = TOGETHER_LLM_MODEL
                api_key = os.environ.get("TOGETHER_API_KEY")

        llm_processor = OpenAILLMProcessor(
            model=llm.model,
            base_url=llm.base_url,
            api_key=api_key,
        )
        return llm_processor

    def get_google_llm_processor(self, llm: LLMConfig) -> LLMProcessor:
        from src.processors.llm.google_llm_processor import GoogleAILLMProcessor

        llm_config = llm
        if llm_config and llm_config.args:
            llm_processor = GoogleAILLMProcessor(**llm_config.args)
        else:
            logging.info("use default google llm processor")
            api_key = os.environ.get("GOOGLE_API_KEY")
            model = os.environ.get("GOOGLE_LLM_MODEL", "gemini-1.5-flash-latest")
            llm_processor = GoogleAILLMProcessor(
                api_key=api_key,
                model=model,
            )

        return llm_processor

    def get_litellm_processor(self, llm: LLMConfig) -> LLMProcessor:
        from src.processors.llm.litellm_processor import LiteLLMProcessor

        llm_processor = LiteLLMProcessor(model=llm.model, set_verbose=False)
        return llm_processor

    def get_llm_processor(self, llm: LLMConfig | None = None) -> LLMProcessor:
        if not llm:
            llm = self._bot_config.llm
        if llm and llm.tag and "vision" in llm.tag:
            # engine llm processor(just support vision model, other TODO):
            # (llm_llamacpp, llm_personalai_proxy, llm_transformers etc..)
            llm_processor = self.get_vision_llm_processor(llm)
        else:
            llm_processor = self.get_remote_llm_processor(llm)
        return llm_processor

    def get_vision_annotate_processor(self) -> AIProcessor:
        from src.processors.vision.annotate_processor import AnnotateProcessor

        detector: IVisionDetector | EngineClass = VisionDetectorEnvInit.initVisionDetectorEngine(
            self._bot_config.vision_detector.tag, self._bot_config.vision_detector.args
        )
        processor = AnnotateProcessor(detector, self.session)
        return processor

    def get_vision_detect_processor(self) -> AIProcessor:
        from src.processors.vision.detect_processor import DetectProcessor

        detector = VisionDetectorEnvInit.initVisionDetectorEngine(
            self._bot_config.vision_detector.tag, self._bot_config.vision_detector.args
        )

        desc, out_desc = "", ""
        if "desc" in self._bot_config.vision_detector.args:
            desc = self._bot_config.vision_detector.args["desc"]
        if "out_desc" in self._bot_config.vision_detector.args:
            out_desc = self._bot_config.vision_detector.args["out_desc"]
        if desc and out_desc:
            processor = DetectProcessor(
                detected_text=desc,
                out_detected_text=out_desc,
                detector=detector,
                session=self.session,
            )
        elif desc:
            processor = DetectProcessor(detected_text=desc, detector=detector, session=self.session)
        elif out_desc:
            processor = DetectProcessor(
                out_detected_text=out_desc, detector=detector, session=self.session
            )
        else:
            processor = DetectProcessor(detector=detector, session=self.session)

        return processor

    def get_vision_ocr_processor(self) -> AIProcessor:
        from src.processors.vision.ocr_processor import OCRProcessor

        ocr = VisionOCREnvInit.initVisionOCREngine(
            self._bot_config.vision_ocr.tag, self._bot_config.vision_ocr.args
        )
        processor = OCRProcessor(ocr=ocr, session=self.session)
        return processor

    def get_moshi_voice_processor(self, llm: LLMConfig | None = None) -> VoiceProcessorBase:
        if not llm:
            llm = self._bot_config.voice_llm
        if llm and llm.tag and "moshi" in llm.tag:
            from src.processors.voice.moshi_voice_processor import (
                MoshiVoiceOpusStreamProcessor,
                MoshiVoiceProcessor,
            )

            if "moshi_opus" in llm.tag:
                if llm.args:
                    llm_processor = MoshiVoiceOpusStreamProcessor(**llm.args)
                else:
                    llm_processor = MoshiVoiceOpusStreamProcessor()
            else:
                if llm.args:
                    llm_processor = MoshiVoiceProcessor(**llm.args)
                else:
                    llm_processor = MoshiVoiceProcessor()
        else:
            from src.processors.voice.moshi_voice_processor import VoiceOpusStreamEchoProcessor

            llm_processor = VoiceOpusStreamEchoProcessor()
        return llm_processor

    def get_text_minicpmo_voice_processor(self, llm: LLMConfig | None = None) -> VoiceProcessorBase:
        from src.processors.voice.minicpmo_voice_processor import MiniCPMoTextVoiceProcessor

        if not llm:
            llm = self._bot_config.voice_llm
        if llm.args:
            llm_processor = MiniCPMoTextVoiceProcessor(**llm.args)
        else:
            llm_processor = MiniCPMoTextVoiceProcessor()
        return llm_processor

    def get_audio_minicpmo_voice_processor(
        self, llm: LLMConfig | None = None
    ) -> VoiceProcessorBase:
        from src.processors.voice.minicpmo_voice_processor import MiniCPMoAudioVoiceProcessor

        if not llm:
            llm = self._bot_config.voice_llm
        if llm.args:
            llm_processor = MiniCPMoAudioVoiceProcessor(**llm.args)
        else:
            llm_processor = MiniCPMoAudioVoiceProcessor()
        return llm_processor

    def get_minicpmo_vision_voice_processor(
        self, llm: LLMConfig | None = None
    ) -> VisionVoiceProcessorBase:
        from src.processors.omni.minicpmo_vision_voice import MiniCPMoVisionVoiceProcessor

        if not llm:
            llm = self._bot_config.omni_llm
        if llm.args:
            llm_processor = MiniCPMoVisionVoiceProcessor(**llm.args)
        else:
            llm_processor = MiniCPMoVisionVoiceProcessor()
        return llm_processor

    def get_text_glm_voice_processor(self, llm: LLMConfig | None = None) -> VoiceProcessorBase:
        from src.processors.voice.glm_voice_processor import GLMTextVoiceProcessor

        if not llm:
            llm = self._bot_config.voice_llm
        if llm.args:
            llm_processor = GLMTextVoiceProcessor(**llm.args)
        else:
            llm_processor = GLMTextVoiceProcessor()
        return llm_processor

    def get_audio_glm_voice_processor(self, llm: LLMConfig | None = None) -> VoiceProcessorBase:
        from src.processors.voice.glm_voice_processor import GLMAudioVoiceProcessor

        if not llm:
            llm = self._bot_config.voice_llm
        if llm.args:
            llm_processor = GLMAudioVoiceProcessor(**llm.args)
        else:
            llm_processor = GLMAudioVoiceProcessor()
        return llm_processor

    def get_text_step_voice_processor(self, llm: LLMConfig | None = None) -> VoiceProcessorBase:
        from src.processors.voice.step_voice_processor import (
            StepTextVoiceProcessor,
            MockStepTextVoiceProcessor,
        )

        if not llm:
            llm = self._bot_config.voice_llm
        if "mock" in llm.tag:
            return MockStepTextVoiceProcessor()
        if llm.args:
            llm_processor = StepTextVoiceProcessor(**llm.args)
        else:
            llm_processor = StepTextVoiceProcessor()
        return llm_processor

    def get_audio_step_voice_processor(self, llm: LLMConfig | None = None) -> VoiceProcessorBase:
        from src.processors.voice.step_voice_processor import (
            StepAudioVoiceProcessor,
            MockStepAudioVoiceProcessor,
        )

        if not llm:
            llm = self._bot_config.voice_llm
        if "mock" in llm.tag:
            return MockStepAudioVoiceProcessor()
        if llm.args:
            llm_processor = StepAudioVoiceProcessor(**llm.args)
        else:
            llm_processor = StepAudioVoiceProcessor()
        return llm_processor

    def get_audio_freeze_omni_voice_processor(
        self, llm: LLMConfig | None = None
    ) -> VoiceProcessorBase:
        cur_dir = os.path.dirname(__file__)
        if bool(os.getenv("ACHATBOT_PKG", "")):
            sys.path.insert(1, os.path.join(cur_dir, "../../FreezeOmni"))
        else:
            sys.path.insert(1, os.path.join(cur_dir, "../../../deps/FreezeOmni"))

        from src.processors.voice.freeze_omni_voice_processor import FreezeOmniVoiceProcessor

        if not llm:
            llm = self._bot_config.voice_llm
        # TODO: need create inference_pipeline_pool and tts pool
        if llm.args:
            llm_processor = FreezeOmniVoiceProcessor(**llm.args)
        else:
            llm_processor = FreezeOmniVoiceProcessor()
        return llm_processor

    def get_openai_voice_processor(self, llm: LLMConfig | None = None) -> VoiceProcessorBase:
        # TODO: use openai reatime voice api
        pass

    def get_tts_processor(self) -> TTSProcessorBase:
        tts_processor: TTSProcessorBase | None = None
        if self._bot_config.tts and self._bot_config.tts.tag and self._bot_config.tts.args:
            if self._bot_config.tts.tag == "elevenlabs_tts_processor":
                from src.processors.speech.tts.elevenlabs_tts_processor import (
                    ElevenLabsTTSProcessor,
                )

                tts_processor = ElevenLabsTTSProcessor(**self._bot_config.tts.args)
            elif self._bot_config.tts.tag == "cartesia_tts_processor":
                from src.processors.speech.tts.cartesia_tts_processor import CartesiaTTSProcessor

                tts_processor = CartesiaTTSProcessor(**self._bot_config.tts.args)
            else:
                # use tts engine processor
                from src.processors.speech.tts.tts_processor import TTSProcessor

                tts = TTSEnvInit.getEngine(self._bot_config.tts.tag, **self._bot_config.tts.args)
                self._bot_config.tts.tag = tts.SELECTED_TAG
                self._bot_config.tts.args = tts.get_args_dict()
                tts_processor = TTSProcessor(tts=tts, session=self.session)
        else:
            # default tts engine processor
            from src.processors.speech.tts.tts_processor import TTSProcessor

            logging.info("use default tts engine processor")
            tag = None
            if self._bot_config.tts and self._bot_config.tts.tag:
                tag = self._bot_config.tts.tag
            tts = TTSEnvInit.initTTSEngine(tag)
            self._bot_config.tts = TTSConfig(tag=tts.SELECTED_TAG, args=tts.get_args_dict())
            tts_processor = TTSProcessor(tts=tts, session=self.session)

        return tts_processor

    def get_image_gen_processor(self) -> ImageGenProcessor:
        if not self._bot_config.img_gen or not self._bot_config.img_gen.args:
            raise Exception("need img_gen args params")
        return get_image_gen_processor(
            self._bot_config.img_gen.tag, **self._bot_config.img_gen.args
        )


class AIRoomBot(AIBot):
    def __init__(self, **args) -> None:
        super().__init__(**args)


class AIChannelBot(AIBot):
    def __init__(self, **args) -> None:
        super().__init__(**args)
