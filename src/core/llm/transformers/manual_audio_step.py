import logging
import os
import sys
from threading import Thread

import torch

from core.llm.transformers.manual_speech_step import TransformersManualSpeechStep
from src.common.utils.helper import get_device
from src.common.session import Session
from src.types.llm.transformers import TransformersLMArgs


class TransformersManualAudioStep(TransformersManualSpeechStep):
    """
    system prompt + (one short: text->speech(audio vq code) prompt) + chat prompt(text/audio) -> tokenizer encode -> token ids -> StepForCausalLM -> audio vq tokens
    with TransformersLMArgs
    """

    TAG = "llm_transformers_manual_audio_step"
    DEFAULT_SYS_PROMPT = "Convert the text to speech"

    def __init__(self, **args):
        super().__init__(**args)

    # @torch.no_grad()
    @torch.inference_mode()
    def generate(self, session: Session, **kwargs):
        """
        system prompt + (one short: text->speech(audio code) prompt) + chat prompt -> tokenizer encode -> token ids -> step lm -> audio vq tokens
        """
