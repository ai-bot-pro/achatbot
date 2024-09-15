from threading import Thread

from .base import TransformersBaseLLM
from src.common.session import Session
from src.types.speech.language import TO_LLM_LANGUAGE


class TransformersManualVisionLLM(TransformersBaseLLM):
    TAG = "llm_transformers_manual_vision"
