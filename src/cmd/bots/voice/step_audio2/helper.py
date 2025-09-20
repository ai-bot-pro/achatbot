import os
import importlib

from src.common.factory import EngineClass
from src.common.interface import ILlm
from src.common.session import Session
from src.types.ai_conf import AIConfig, LLMConfig, BaseConfig
from src.common.types import MODELS_DIR, ASSETS_DIR
from src.processors.voice.step_audio2_processor import Token2wav, StepAudio2BaseProcessor

from src.core.llm import LLMEnvInit


def get_token2wav(llm_config: BaseConfig):
    lm_model_name_or_path = llm_config.args.get(
        "lm_model_name_or_path", os.path.join(MODELS_DIR, "stepfun-ai/Step-Audio-2-mini")
    )
    token2wav_path = os.path.join(lm_model_name_or_path, "token2wav")
    return Token2wav(
        token2wav_path,
        **llm_config.args,
    )


def get_step_audio2_llm(
    llm_config: BaseConfig,
) -> ILlm | EngineClass:
    tag = llm_config.tag if llm_config.tag else "llm_transformers_manual_voice_step2"
    return LLMEnvInit.initLLMEngine(tag, llm_config.args)


def get_step_audio2_transformers_llm(llm_config: BaseConfig):
    from src.core.llm.transformers.manual_voice_step2 import TransformersManualVoiceStep2

    lm_model_name_or_path = os.path.join(MODELS_DIR, "stepfun-ai/Step-Audio-2-mini")
    args = llm_config.args if llm_config.args else {}
    if args.get("lm_model_name_or_path", None) is None:
        args["lm_model_name_or_path"] = lm_model_name_or_path
    return TransformersManualVoiceStep2(**args)


def get_step_audio2_vllm_client_llm(llm_config: BaseConfig):
    from src.core.llm.vllm.step_audio2 import VllmClientStepAudio2

    return VllmClientStepAudio2(**llm_config.args)


def get_step_audio2_processor(
    llm_config: BaseConfig,
    session: Session | None = None,
    token2wav: Token2wav | None = None,
    audio_llm: ILlm | None = None,
    processor_class_name: str | None = None,
) -> StepAudio2BaseProcessor:
    if processor_class_name is None and hasattr(llm_config, "processor"):
        processor_class_name = llm_config.processor
    try:
        if bool(os.getenv("ACHATBOT_PKG", "")):
            module = importlib.import_module("achatbot.processors.voice.step_audio2_processor")
        else:
            module = importlib.import_module("src.processors.voice.step_audio2_processor")
        processor_class = getattr(module, processor_class_name)
        return processor_class(
            session=session,
            token2wav=token2wav,
            audio_llm=audio_llm or get_step_audio2_llm(llm_config),
            **llm_config.args,
        )
    except (ImportError, AttributeError) as e:
        raise ValueError(f"cannot import {processor_class_name}: {str(e)}")


"""
python -m src.cmd.bots.voice.step_audio2.helper
ACHATBOT_PKG=1 python -m src.cmd.bots.voice.step_audio2.helper
"""
if __name__ == "__main__":
    get_step_audio2_processor(
        LLMConfig(
            processor="StepAudio2TextAudioChatProcessor",
            args={
                "init_system_prompt": "",
                "prompt_wav": "/root/.achatbot/assets/default_male.wav",
                "warmup_cn": 2,
                "chat_history_size": None,
                "text_stream_out": False,
                "no_stream_sleep_time": 0.5,
                "lm_model_name_or_path": "stepfun-ai/Step-Audio-2-mini",
                "lm_gen_max_new_tokens": 64,
                "lm_gen_temperature": 0.1,
                "lm_gen_top_k": 20,
                "lm_gen_top_p": 0.95,
                "repetition_penalty": 1.1,
            },
        )
    )
