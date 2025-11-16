import os
import logging
import asyncio
import sys
import uuid
import signal
from typing import Dict, Type, Any, Optional

from apipeline.frames import CancelFrame
from apipeline.pipeline.task import PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from dotenv import load_dotenv
import nest_asyncio

from src.processors.context.memory import MemoryProcessor
from src.processors.omni.base import VisionVoiceProcessorBase
from src.processors.voice.base import VoiceProcessorBase
from src.processors.image.base import ImageGenProcessor
from src.processors.image import get_image_gen_processor
from src.modules.vision.ocr import VisionOCREnvInit
from src.modules.vision.detector import VisionDetectorEnvInit
from src.processors.ai_processor import AIProcessor, FrameProcessor
from src.processors.vision.vision_processor import MockVisionProcessor
from src.processors.avatar.base import AvatarProcessorBase
from src.processors.speech.asr.base import ASRProcessorBase
from src.processors.llm.base import LLMProcessor
from src.processors.speech.tts.base import TTSProcessorBase
from src.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from src.modules.speech.asr import ASREnvInit
from src.core.llm import LLMEnvInit
from src.modules.speech.tts import TTSEnvInit
from src.modules.avatar import AvatarEnvInit
from src.types.ai_conf import (
    BaseConfig,
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
from src.common.logger import Logger
from src.common.pool_engine import EngineProviderPool, PoolInstanceInfo
from src.processors.session_processor import SessionProcessor

load_dotenv(override=True)

use_nest_asyncio = os.getenv("NEST_ASYNCIO", "1") != "0"
if use_nest_asyncio:
    nest_asyncio.apply()


Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)


class AIBot(IBot):
    r"""
    use ai bot config
    !TODONE: need config processor with bot config (redefine api params) @weedge
    bot config: Dict[str, Dict[str,Any]]
    e.g. {"llm":{"key":val,"tag":TAG,"args":{}}, "tts":{"key":val,"tag":TAG,"args":{}}}
    !TIPS: RTVI config options can transfer to ai bot config

    !NOTE:
    use multiprocessing pipe to run bot with unix socket, bot __init__ to new a bot obj must be serializable (pickle); or wraper a func, don't use bot obj method.
    """

    def __init__(self, **args) -> None:
        self.args = BotRunArgs(**args)
        if self.args.bot_name is None or len(self.args.bot_name) == 0:
            self.args.bot_name = self.__class__.__name__

        self.task: PipelineTask | None = None
        self.session = Session(**SessionCtx(str(uuid.uuid4())).__dict__)
        self.runner: PipelineRunner | None = None
        self.generator: interface.ILlmGenerator = None

        self._bot_config_list = self.args.bot_config_list
        self._bot_config = self.args.bot_config
        self._handle_sigint = self.args.handle_sigint
        self._save_audio = self.args.save_audio

        self.vad_analyzer_pool: EngineProviderPool = None
        self.asr_pool: EngineProviderPool = None
        self.translate_llm_generator_pool: EngineProviderPool = None
        self.tts_pool: EngineProviderPool = None
        self.avatar_pool: EngineProviderPool = None

    def init_bot_config(self):
        try:
            logging.debug(f"args.bot_config: {self.args.bot_config}")
            self._bot_config: AIConfig = AIConfig(**self.args.bot_config)
            if len(self._bot_config_list) > 0:
                from src.types.rtvi import RTVIConfig

                rtvi_config = RTVIConfig(config_list=self.args.bot_config_list)
                self._bot_config = AIConfig(**rtvi_config._arguments_dict)
            if self._bot_config.llm is None:
                self._bot_config.llm = LLMConfig()
        except Exception as e:
            raise Exception(f"Failed to parse bot configuration: {e}")
        logging.info(f"ai bot_config: {self._bot_config}")

    def set_args(self, args):
        merge_args = {**self.args.__dict__, **args}
        self.args = BotRunArgs(**merge_args)

    def bot_config(self):
        return self._bot_config

    def load(self):
        """
        NOTE:
        - init modules here
            # load model ckpt when bot start
            # when deploy need load model ckpt, then run serve
        - don't init processor; processor for each connect session
        - load model engine here to share,
        - if don't need share, init in processor, u can init a engine pool
        have fun :)
        """
        pass

    def run(self):
        asyncio.run(self.async_run())

    def __del__(self):
        self.cancel()

    def cancel(self):
        if self.runner:
            asyncio.create_task(self.runner.cancel())
        if self.generator:
            self.generator.close()

    async def async_run(self):
        try:
            await self.arun()
        except asyncio.CancelledError:
            logging.info("CancelledError, Exiting!")
            if self.task is not None:
                logging.info("Cancelling task!")
                await self.task.queue_frame(CancelFrame())
        except KeyboardInterrupt:
            logging.warning("Ctrl-C detected. Exiting!")
            if self.task is not None:
                logging.info("Cancelling task!")
                await self.task.queue_frame(CancelFrame())
        except Exception as e:
            logging.error(f"run error: {e}", exc_info=True)
        finally:
            if self.args.handle_sigint is True:
                loop = asyncio.get_running_loop()
                loop.remove_signal_handler(signal.SIGINT)
                loop.remove_signal_handler(signal.SIGTERM)
            self.cancel()
            logging.info(f"{__name__} task run is Done!")

    async def arun(self):
        pass

    def get_pool(self, conf: BaseConfig, creation_func: callable) -> EngineProviderPool | None:
        pool: EngineProviderPool = None
        if conf and conf.pool_size:
            pool = EngineProviderPool(
                pool_size=conf.pool_size,
                new_func=creation_func,
                init_worker_num=conf.pool_init_worker_num,
            )

        if pool and not pool.initialize():
            logging.error(f"Failed to initialize pool func:{creation_func} with conf:{conf}")
            return None

        return pool

    def get_vad_analyzer_pool(self):
        return self.get_pool(self._bot_config.vad, self.get_vad_analyzer)

    def get_vad_analyzer_from_pool(
        self,
    ) -> tuple[Optional[PoolInstanceInfo], interface.IVADAnalyzer | EngineClass]:
        vad_analyzer_info = None
        if self.vad_analyzer_pool:
            vad_analyzer_info = self.vad_analyzer_pool.get()
        vad_analyzer = (
            vad_analyzer_info.get_instance() if vad_analyzer_info else self.get_vad_analyzer()
        )
        return (vad_analyzer_info, vad_analyzer)

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

    def get_avatar_pool(self):
        return self.get_pool(self._bot_config.avatar, self.get_avatar)

    def get_avatar_from_pool(self) -> tuple[Optional[PoolInstanceInfo], EngineClass | None]:
        avatar_info = None
        if self.avatar_pool:
            avatar_info = self.avatar_pool.get()
        avatar = avatar_info.get_instance() if avatar_info else self.get_avatar()
        return (avatar_info, avatar)

    def get_avatar(self) -> EngineClass | None:
        avatar: EngineClass | None = None
        if self._bot_config.avatar and self._bot_config.avatar.tag:
            if self._bot_config.avatar.tag == "lam_audio2expression_avatar":
                from src.modules.avatar.lam_audio2expression import LAMAudio2ExpressionAvatar

                if self._bot_config.avatar.args:
                    avatar = LAMAudio2ExpressionAvatar(**self._bot_config.avatar.args)
                else:
                    avatar = LAMAudio2ExpressionAvatar()
            else:
                # TODO: use avatar engine
                args = self._bot_config.avatar.args or {}
                avatar = AvatarEnvInit.getEngine(self._bot_config.avatar.tag, **args)
                raise NotImplementedError("don't support use tag to create avatar engine")
            avatar.load()
            return avatar
        else:
            from src.modules.avatar.lite_avatar import LiteAvatar

            logging.info("use default lite avatar engine processor")
            if self._bot_config.avatar and self._bot_config.avatar.args:
                avatar = LiteAvatar(**self._bot_config.avatar.args)
            else:
                avatar = LiteAvatar()
            avatar.load()
        return avatar

    def get_a2a_processor(self) -> SessionProcessor:
        processor: SessionProcessor | None = None
        if self._bot_config.a2a and self._bot_config.a2a.args:
            from src.processors.a2a.a2a_conversation_processor import A2AConversationProcessor

            processor = A2AConversationProcessor(session=self.session, **self._bot_config.a2a.args)
            return processor

        raise ValueError("A2A processor is not configured. Please check your bot configuration.")

    def get_avatar_processor(self, avatar=None) -> AvatarProcessorBase:
        avatar_processor: AvatarProcessorBase | None = None
        # use avatar engine processor

        if self._bot_config.avatar and self._bot_config.avatar.tag:
            if self._bot_config.avatar.tag == "lam_audio2expression_avatar":
                from src.processors.avatar.lam_audio2expression_avatar_processor import (
                    LAMAudio2ExpressionAvatarProcessor,
                )
                from src.modules.avatar.lam_audio2expression import LAMAudio2ExpressionAvatar

                if self._bot_config.avatar and self._bot_config.avatar.args:
                    avatar = avatar or LAMAudio2ExpressionAvatar(**self._bot_config.avatar.args)
                else:
                    avatar = avatar or LAMAudio2ExpressionAvatar()
                return LAMAudio2ExpressionAvatarProcessor(avatar, **self._bot_config.avatar.args)
            else:
                # TODO: use avatar engine processor
                args = self._bot_config.avatar.args or {}
                avatar = AvatarEnvInit.getEngine(self._bot_config.avatar.tag, **args)
                _ = avatar
                raise NotImplementedError("don't support use tag to create avatar engine")
                return avatar_processor
        else:
            from src.processors.avatar.lite_avatar_processor import LiteAvatarProcessor
            from src.modules.avatar.lite_avatar import LiteAvatar

            logging.info("use default lite avatar engine processor")
            if self._bot_config.avatar and self._bot_config.avatar.args:
                avatar = avatar or LiteAvatar(**self._bot_config.avatar.args)
            else:
                avatar = avatar or LiteAvatar()
            return LiteAvatarProcessor(avatar)

    def get_asr_pool(self):
        return self.get_pool(self._bot_config.asr, self.get_asr)

    def get_asr_from_pool(
        self,
    ) -> tuple[Optional[PoolInstanceInfo], interface.IAsr | EngineClass | None]:
        asr_info = None
        if self.asr_pool:
            asr_info = self.asr_pool.get()
        asr = asr_info.get_instance() if asr_info else self.get_asr()
        return (asr_info, asr)

    def get_asr(self) -> interface.IAsr | EngineClass | None:
        asr: interface.IAsr | EngineClass | None = None
        if (
            self._bot_config.asr
            and self._bot_config.asr.tag
            and self._bot_config.asr.tag == "deepgram_asr_processor"
            and self._bot_config.asr.args
        ):
            pass
        else:
            if self._bot_config.asr and self._bot_config.asr.tag and self._bot_config.asr.args:
                asr = ASREnvInit.getEngine(self._bot_config.asr.tag, **self._bot_config.asr.args)
            else:
                logging.info("use default asr engine processor")
                asr = ASREnvInit.initASREngine()
        return asr

    def get_asr_processor(
        self, asr_engine: interface.IAsr | EngineClass | None = None
    ) -> ASRProcessorBase:
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

            if self._bot_config.asr and self._bot_config.asr.tag and self._bot_config.asr.args:
                asr = asr_engine or ASREnvInit.getEngine(
                    self._bot_config.asr.tag, **self._bot_config.asr.args
                )
            else:
                logging.info("use default asr engine processor")
                asr = asr_engine or ASREnvInit.initASREngine()
                self._bot_config.asr = ASRConfig(tag=asr.SELECTED_TAG, args=asr.get_args_dict())
            asr_processor = ASRProcessor(asr=asr, session=self.session)
        return asr_processor

    def get_hf_tokenizer(self):
        """
        To disable this warning, you can either:
           - Avoid using `tokenizers` before the fork if possible
           - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self._bot_config.translate_llm.model)
        return tokenizer

    def get_translate_llm_generator_pool(self):
        return self.get_pool(self._bot_config.translate_llm, self.get_translate_llm_generator)

    def get_translate_llm_generator_from_pool(
        self,
    ) -> tuple[
        Optional[PoolInstanceInfo], interface.ILlmGenerator | interface.ILlm | EngineClass | None
    ]:
        translate_llm_generator_info = None
        if self.translate_llm_generator_pool:
            translate_llm_generator_info = self.translate_llm_generator_pool.get()
        translate_llm_generator = (
            translate_llm_generator_info.get_instance()
            if translate_llm_generator_info
            else self.get_translate_llm_generator()
        )
        return (translate_llm_generator_info, translate_llm_generator)

    def get_translate_llm_generator(self):
        """
        get local translate llm generator
        """
        assert self._bot_config.translate_llm, f"translate_llm must be provided {self._bot_config=}"
        # load llm generator
        tag = self._bot_config.translate_llm.tag
        args = self._bot_config.translate_llm.args or {}
        generator = LLMEnvInit.initLLMEngine(
            tag=tag,
            kwargs=args,
        )
        return generator

    def get_vision_llm_processor(
        self, llm_config: LLMConfig | None = None, llm_engine=None
    ) -> LLMProcessor:
        """
        get local vision llm
        """
        from src.processors.vision.vision_processor import VisionProcessor

        if not llm_config:
            llm_config = self._bot_config.vision_llm
        if "mock" in llm_config.tag:
            llm_processor = MockVisionProcessor()
        else:
            logging.info(f"init engine llm processor tag: {llm_config.tag}")
            sleep_time_s = llm_config.args.pop("sleep_time_s", 0.15)
            llm_engine = llm_engine or LLMEnvInit.initLLMEngine(llm_config.tag, llm_config.args)
            llm_processor = VisionProcessor(llm_engine, self.session, sleep_time_s=sleep_time_s)
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
            elif "local" in llm.base_url or "127.0.0.1" in llm.base_url:
                api_key = "ollama"

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
            logging.info(f"use google llm processor args:{llm_config.args}")
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

    def get_llm_processor(self, llm: LLMConfig | None = None, llm_engine=None) -> LLMProcessor:
        if not llm:
            llm = self._bot_config.llm or self._bot_config.vision_llm
        if llm and llm.tag and "vision" in llm.tag:
            # engine llm processor(just support vision model, other TODO):
            # (llm_llamacpp, llm_personalai_proxy, llm_transformers etc..)
            llm_processor = self.get_vision_llm_processor(llm, llm_engine)
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

    def get_text_qwen2_5omni_voice_processor(
        self, llm: LLMConfig | None = None
    ) -> VoiceProcessorBase:
        from src.processors.voice.qwen2_5omni_voice_processor import Qwen2_5OmniTextVoiceProcessor

        if not llm:
            llm = self._bot_config.voice_llm
        if llm.args:
            llm_processor = Qwen2_5OmniTextVoiceProcessor(**llm.args)
        else:
            llm_processor = Qwen2_5OmniTextVoiceProcessor()
        return llm_processor

    def get_text_kimi_voice_processor(self, llm: LLMConfig | None = None) -> VoiceProcessorBase:
        from src.processors.voice.kimi_voice_processor import KimiTextVoiceProcessor

        if not llm:
            llm = self._bot_config.voice_llm
        if llm.args:
            llm_processor = KimiTextVoiceProcessor(**llm.args)
        else:
            llm_processor = KimiTextVoiceProcessor()
        return llm_processor

    def get_text_vita_voice_processor(self, llm: LLMConfig | None = None) -> VoiceProcessorBase:
        from src.processors.voice.vita_voice_processor import VITATextVoiceProcessor

        if not llm:
            llm = self._bot_config.voice_llm
        if llm.args:
            llm_processor = VITATextVoiceProcessor(**llm.args)
        else:
            llm_processor = VITATextVoiceProcessor()
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

    def get_audio_qwen2_5omni_voice_processor(
        self, llm: LLMConfig | None = None
    ) -> VoiceProcessorBase:
        from src.processors.voice.qwen2_5omni_voice_processor import Qwen2_5OmniAudioVoiceProcessor

        if not llm:
            llm = self._bot_config.voice_llm
        if llm.args:
            llm_processor = Qwen2_5OmniAudioVoiceProcessor(**llm.args)
        else:
            llm_processor = Qwen2_5OmniAudioVoiceProcessor()
        return llm_processor

    def get_audio_kimi_voice_processor(self, llm: LLMConfig | None = None) -> VoiceProcessorBase:
        from src.processors.voice.kimi_voice_processor import KimiAudioVoiceProcessor

        if not llm:
            llm = self._bot_config.voice_llm
        if llm.args:
            llm_processor = KimiAudioVoiceProcessor(**llm.args)
        else:
            llm_processor = KimiAudioVoiceProcessor()
        return llm_processor

    def get_audio_vita_voice_processor(self, llm: LLMConfig | None = None) -> VoiceProcessorBase:
        from src.processors.voice.vita_voice_processor import VITAAudioVoiceProcessor

        if not llm:
            llm = self._bot_config.voice_llm
        if llm.args:
            llm_processor = VITAAudioVoiceProcessor(**llm.args)
        else:
            llm_processor = VITAAudioVoiceProcessor()
        return llm_processor

    def get_audio_phi4_speech_processor(self, llm: LLMConfig | None = None) -> VoiceProcessorBase:
        from src.processors.voice.phi4_speech_processor import Phi4AudioTextProcessor

        if not llm:
            llm = self._bot_config.voice_llm
        if llm.args:
            llm_processor = Phi4AudioTextProcessor(**llm.args)
        else:
            llm_processor = Phi4AudioTextProcessor()
        return llm_processor

    def get_minicpmo_vision_voice_processor(
        self, llm: LLMConfig | None = None
    ) -> VisionVoiceProcessorBase:
        from src.processors.omni.minicpmo_vision_voice import MiniCPMoVisionVoiceProcessor
        from src.processors.omni.base import MockVisionVoiceProcessor

        if not llm:
            llm = self._bot_config.omni_llm
        if "mock" in llm.tag:
            return MockVisionVoiceProcessor()
        if llm.args:
            llm_processor = MiniCPMoVisionVoiceProcessor(**llm.args)
        else:
            llm_processor = MiniCPMoVisionVoiceProcessor()
        return llm_processor

    def get_qwen2_5omni_vision_voice_processor(
        self, llm: LLMConfig | None = None
    ) -> VisionVoiceProcessorBase:
        from src.processors.omni.qwen2_5omni_vision_voice import Qwen2_5OmnVisionVoiceProcessor
        from src.processors.omni.base import MockVisionVoiceProcessor

        if not llm:
            llm = self._bot_config.omni_llm
        if "mock" in llm.tag:
            return MockVisionVoiceProcessor()
        if llm.args:
            llm_processor = Qwen2_5OmnVisionVoiceProcessor(**llm.args)
        else:
            llm_processor = Qwen2_5OmnVisionVoiceProcessor()
        return llm_processor

    def get_phi4_vision_speech_processor(
        self, llm: LLMConfig | None = None
    ) -> VisionVoiceProcessorBase:
        from src.processors.omni.phi4_vision_speech import Phi4VisionSpeechProcessor
        from src.processors.omni.base import MockVisionVoiceProcessor

        if not llm:
            llm = self._bot_config.omni_llm
        if "mock" in llm.tag:
            return MockVisionVoiceProcessor()
        if llm.args:
            llm_processor = Phi4VisionSpeechProcessor(**llm.args)
        else:
            llm_processor = Phi4VisionSpeechProcessor()
        return llm_processor

    def get_gemma3n_vision_speech_processor(
        self, llm: LLMConfig | None = None
    ) -> VisionVoiceProcessorBase:
        from src.processors.omni.gemma_vision_speech import Gemma3nVisionSpeechProcessor
        from src.processors.omni.base import MockVisionVoiceProcessor

        if not llm:
            llm = self._bot_config.omni_llm
        if "mock" in llm.tag:
            return MockVisionVoiceProcessor()
        if llm.args:
            llm_processor = Gemma3nVisionSpeechProcessor(**llm.args)
        else:
            llm_processor = Gemma3nVisionSpeechProcessor()
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

    def get_tts_pool(self):
        return self.get_pool(self._bot_config.tts, self.get_tts)

    def get_tts_from_pool(
        self,
    ) -> tuple[Optional[PoolInstanceInfo], interface.ITts | EngineClass | None]:
        tts_info = None
        if self.tts_pool:
            tts_info = self.tts_pool.get()
        tts = tts_info.get_instance() if tts_info else self.get_tts()
        return (tts_info, tts)

    def get_tts(self) -> interface.ITts | EngineClass | None:
        tts: interface.ITts | EngineClass | None = None
        if (
            self._bot_config.tts
            and self._bot_config.tts.tag
            and self._bot_config.tts.tag in ["elevenlabs_tts_processor", "cartesia_tts_processor"]
        ):
            pass
        else:
            if self._bot_config.tts and self._bot_config.tts.tag and self._bot_config.tts.args:
                tts = TTSEnvInit.getEngine(self._bot_config.tts.tag, **self._bot_config.tts.args)
            else:
                logging.info("use default tts engine processor")
                tts = TTSEnvInit.initttsEngine()
        return tts

    def get_tts_processor(
        self, tts_engine: interface.ITts | EngineClass = None
    ) -> TTSProcessorBase:
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

                tts_engine = tts_engine or TTSEnvInit.getEngine(
                    self._bot_config.tts.tag, **self._bot_config.tts.args
                )
                self._bot_config.tts.tag = tts_engine.SELECTED_TAG
                self._bot_config.tts.args = tts_engine.get_args_dict()
                tts_processor = TTSProcessor(
                    tts=tts_engine,
                    session=self.session,
                    aggregate_sentences=self._bot_config.tts.aggregate_sentences,
                    push_text_frames=self._bot_config.tts.push_text_frames,
                    remove_punctuation=self._bot_config.tts.remove_punctuation,
                )
        else:
            # default tts engine processor
            from src.processors.speech.tts.tts_processor import TTSProcessor

            logging.info("use default tts engine processor")
            tag = None
            if self._bot_config.tts and self._bot_config.tts.tag:
                tag = self._bot_config.tts.tag
            tts_engine = tts_engine or TTSEnvInit.initTTSEngine(tag)
            self._bot_config.tts = TTSConfig(
                tag=tts_engine.SELECTED_TAG, args=tts_engine.get_args_dict()
            )
            tts_processor = TTSProcessor(
                tts=tts_engine,
                session=self.session,
                aggregate_sentences=self._bot_config.tts.aggregate_sentences,
                push_text_frames=self._bot_config.tts.push_text_frames,
                remove_punctuation=self._bot_config.tts.remove_punctuation,
            )

        return tts_processor

    def get_image_gen_processor(self) -> ImageGenProcessor:
        if not self._bot_config.img_gen or not self._bot_config.img_gen.args:
            raise Exception("need img_gen args params")
        return get_image_gen_processor(
            self._bot_config.img_gen.tag, **self._bot_config.img_gen.args
        )

    def get_memory_processor(
        self,
        local_config: Optional[Dict[str, Any]] = None,
    ) -> MemoryProcessor:
        if not self._bot_config.memory or not self._bot_config.memory.processor:
            raise ValueError("Memory processor configuration is missing or invalid.")

        processor_type = self._bot_config.memory.processor
        args = self._bot_config.memory.args or {}

        if processor_type == "Mem0MemoryProcessor":
            from src.processors.context.memory.mem0 import Mem0MemoryProcessor

            return Mem0MemoryProcessor(local_config=local_config, **args)

        raise NotImplementedError(f"Memory processor '{processor_type}' is not supported.")


class AIRoomBot(AIBot):
    def __init__(self, **args) -> None:
        super().__init__(**args)


class AIChannelBot(AIBot):
    def __init__(self, **args) -> None:
        super().__init__(**args)
