import os
import importlib

from src.common.interface import ILlm
from src.common.session import Session
from src.types.ai_conf import AIConfig, LLMConfig
from src.common.types import MODELS_DIR
from src.processors.voice.step_audio2_processor import Token2wav


def get_step_audio2_llm(bot_config: AIConfig):
    from src.core.llm.transformers.manual_voice_step2 import TransformersManualVoiceStep2

    llm = bot_config.llm or bot_config.voice_llm
    lm_model_name_or_path = os.path.join(MODELS_DIR, "stepfun-ai/Step-Audio-2-mini")
    if llm.args and llm.args.get("lm_model_name_or_path"):
        lm_model_name_or_path = llm.args.get("lm_model_name_or_path")
    return TransformersManualVoiceStep2(lm_model_name_or_path=lm_model_name_or_path)


def get_step_audio2_processor(
    bot_config: AIConfig,
    session: Session | None = None,
    token2wav: Token2wav | None = None,
    audio_llm: ILlm | None = None,
):
    llm_conf = bot_config.voice_llm or bot_config.llm
    if not llm_conf:
        raise ValueError("llm confk is None")
    try:
        if bool(os.getenv("ACHATBOT_PKG", "")):
            module = importlib.import_module("achatbot.processors.voice.step_audio2_processor")
        else:
            module = importlib.import_module("src.processors.voice.step_audio2_processor")
        processor_class = getattr(module, llm_conf.processor)
        return processor_class(
            session=session,
            token2wav=token2wav,
            audio_llm=audio_llm or get_step_audio2_llm(bot_config),
            **llm_conf.args,
        )
    except (ImportError, AttributeError) as e:
        raise ValueError(f"cannot import {llm_conf.processor}: {str(e)}")


"""
python -m src.cmd.bots.voice.step_audio2.helper
ACHATBOT_PKG=1 python -m src.cmd.bots.voice.step_audio2.helper
"""
if __name__ == "__main__":
    get_step_audio2_processor(
        AIConfig(
            voice_llm=LLMConfig(
                processor="StepAudioText2TextProcessor",
                args={"lm_model_name_or_path": "stepfun-ai/Step-Audio-2-mini"},
            )
        ),
    )
