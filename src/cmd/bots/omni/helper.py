import os
import importlib

from src.common.factory import EngineClass
from src.common.interface import ILlm
from src.common.session import Session
from src.types.ai_conf import BaseConfig
from src.processors.omni.base import VisionVoiceProcessorBase

from src.core.llm import LLMEnvInit


def get_qwen3omni_llm(
    llm_config: BaseConfig,
) -> ILlm | EngineClass:
    tag = llm_config.tag if llm_config.tag else "llm_transformers_manual_qwen3omni_vision_voice"
    return LLMEnvInit.initLLMEngine(tag, llm_config.args)


def get_qwen3omni_processor(
    llm_config: BaseConfig,
    llm: ILlm | None = None,
    session: Session | None = None,
    processor_class_name: str | None = None,
) -> VisionVoiceProcessorBase:
    if processor_class_name is None and hasattr(llm_config, "processor"):
        processor_class_name = llm_config.processor
    try:
        omni_module_name = "base"
        if hasattr(llm_config, "processor") and "Mock" not in llm_config.processor:
            omni_module_name = "qwen3omni_vision_voice"
        if bool(os.getenv("ACHATBOT_PKG", "")):
            module = importlib.import_module(f"achatbot.processors.omni.{omni_module_name}")
        else:
            module = importlib.import_module(f"src.processors.omni.{omni_module_name}")
        processor_class = getattr(module, processor_class_name)
        return processor_class(
            llm=llm or get_qwen3omni_llm(llm_config),
            session=session,
            **llm_config.args,
        )
    except (ImportError, AttributeError) as e:
        raise ValueError(f"cannot import {processor_class_name}: {str(e)}")
