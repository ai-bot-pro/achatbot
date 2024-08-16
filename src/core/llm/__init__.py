import logging
import os

from src.common.types import (
    DEFAULT_SYSTEM_PROMPT, MODELS_DIR,
    PersonalAIProxyArgs)
from src.common import interface
from src.common.factory import EngineClass, EngineFactory

from dotenv import load_dotenv
load_dotenv(override=True)


class LLMEnvInit():

    @staticmethod
    def getEngine(tag, **kwargs) -> interface.IAsr | EngineClass:
        if "llm_llamacpp" in tag:
            from . import llamacpp
        elif "llm_personalai_proxy" in tag:
            from . import personalai

        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        return engine

    @staticmethod
    def initLLMEngine() -> interface.ILlm | EngineClass:
        # llm
        tag = os.getenv('LLM_TAG', "llm_llamacpp")
        kwargs = LLMEnvInit.map_config_func[tag]()
        engine = LLMEnvInit.getEngine(tag, **kwargs)
        logging.info(f"initLLMEngine: {tag}, {engine}")
        return engine

    @staticmethod
    def get_llm_llamacpp_args() -> dict:
        kwargs = {}
        kwargs["model_name"] = os.getenv('LLM_MODEL_NAME', 'qwen2')
        kwargs["model_path"] = os.getenv('LLM_MODEL_PATH', os.path.join(
            MODELS_DIR, "qwen2-1_5b-instruct-q8_0.gguf"))
        kwargs["model_type"] = os.getenv('LLM_MODEL_TYPE', "chat")
        kwargs["n_threads"] = os.cpu_count()
        kwargs["verbose"] = False
        kwargs["n_gpu_layers"] = int(os.getenv('N_GPU_LAYERS', "0"))
        kwargs["flash_attn"] = bool(os.getenv('FLASH_ATTN', ""))
        kwargs["llm_stream"] = True
        # if logger.getEffectiveLevel() != logging.DEBUG:
        #    kwargs["verbose"] = False
        kwargs['llm_stop'] = [
            "<|end|>", "<|im_end|>", "<|endoftext|>", "<{/end}>",
            "</s>", "/s>", "</s", "<s>",
            "<|user|>", "<|assistant|>", "<|system|>",
        ]
        kwargs["llm_chat_system"] = os.getenv('LLM_CHAT_SYSTEM', DEFAULT_SYSTEM_PROMPT)
        kwargs["llm_tool_choice"] = os.getenv('LLM_TOOL_CHOICE', None)
        kwargs["chat_format"] = os.getenv('LLM_CHAT_FORMAT', None)
        kwargs["tokenizer_path"] = os.getenv('LLM_TOKENIZER_PATH', None)
        return kwargs

    @staticmethod
    def get_llm_personal_ai_proxy_args() -> dict:
        kwargs = PersonalAIProxyArgs(
            api_url=os.getenv('API_URL', "http://localhost:8787/"),
            chat_bot=os.getenv('CHAT_BOT', "openai"),
            model_type=os.getenv('CHAT_TYPE', "chat_only"),
            openai_api_base_url=os.getenv('OPENAI_API_BASE_URL', "https://api.groq.com/openai/v1/"),
            model_name=os.getenv('LLM_MODEL_NAME', "llama3-70b-8192"),
            llm_chat_system=os.getenv('LLM_CHAT_SYSTEM', DEFAULT_SYSTEM_PROMPT),
            llm_stream=False,
            llm_max_tokens=int(os.getenv('LLM_MAX_TOKENS', "1024")),
            func_search_name=os.getenv('FUNC_SEARCH_NAME', "search_api"),
            func_weather_name=os.getenv('FUNC_WEATHER_NAME', "openweathermap_api"),
        ).__dict__
        return kwargs

    # TAG : config
    map_config_func = {
        'llm_llamacpp': get_llm_llamacpp_args,
        'llm_personalai_proxy': get_llm_personal_ai_proxy_args,
    }
