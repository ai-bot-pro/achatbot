import logging

from src.core.llm.transformers.base import TransformersBaseLLM
from src.types.llm.transformers import TransformersLMArgs


class TransformersManualVoicFreezeOmni(TransformersBaseLLM):
    """ """

    TAG = "llm_transformers_manual_voice_freeze_omni"
    DEFAULT_SYS_PROMPT = "You are a helpful assistant."

    def __init__(self, **args):
        self.args = TransformersLMArgs(**args)
        logging.info("TransformersManualVoiceFreezeOmni args: %s", self.args)
