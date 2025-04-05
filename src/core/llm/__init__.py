import json
import logging
import os

from dotenv import load_dotenv

from src.common.types import DEFAULT_SYSTEM_PROMPT, MODELS_DIR, PersonalAIProxyArgs
from src.common import interface
from src.common.factory import EngineClass, EngineFactory


load_dotenv(override=True)


class LLMEnvInit:
    @staticmethod
    def getEngine(tag, **kwargs) -> interface.ILlm | EngineClass:
        if "llm_llamacpp" == tag:
            from . import llamacpp
        if "llm_llamacpp_generator" == tag:
            from .llamacpp.generator import LlamacppGenerator
        if "llm_vllm_generator" == tag:
            from .vllm.generator import VllmGenerator
        if "llm_trtllm_generator" == tag:
            from .tensorrt_llm.generator import TrtLLMGenerator
        if "llm_trtllm_runner_generator" == tag:
            from .tensorrt_llm.generator import TrtLLMRunnerGenerator
        if "llm_sglang_generator" == tag:
            from .sglang.generator import SGlangGenerator
        elif "llm_personalai_proxy" == tag:
            from . import personalai
        if "llm_transformers_generator" == tag:
            from .transformers.generator import TransformersGenerator
        elif "llm_transformers_manual_vision_qwen" == tag:
            from .transformers import manual_vision_qwen
        elif "llm_transformers_manual_vision_llama" == tag:
            from .transformers import manual_vision_llama
        elif "llm_transformers_manual_vision_molmo" == tag:
            from .transformers import manual_vision_molmo
        elif "llm_transformers_manual_vision_deepseek" == tag:
            from .transformers import manual_vision_deepseek
        elif "llm_transformers_manual_vision_janus_flow" == tag:
            from .transformers import manual_vision_img_janus_flow
        elif "llm_transformers_manual_image_janus_flow" == tag:
            from .transformers import manual_vision_img_janus_flow
        elif "llm_transformers_manual_vision_janus" == tag:
            from .transformers import manual_vision_img_janus_pro
        elif "llm_transformers_manual_image_janus" == tag:
            from .transformers import manual_vision_img_janus_pro
        elif "llm_transformers_manual_vision_minicpmo" == tag:
            from .transformers import manual_vision_voice_minicpmo
        elif "llm_transformers_manual" == tag:
            from .transformers import manual
        elif "llm_transformers_pipeline" == tag:
            from .transformers import pipeline

        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        return engine

    @staticmethod
    def initLLMEngine(
        tag: str | None = None, kwargs: dict | None = None
    ) -> interface.ILlmGenerator | interface.ILlm | EngineClass:
        # llm
        tag = tag if tag else os.getenv("LLM_TAG", "llm_llamacpp")
        if kwargs is None:
            kwargs = LLMEnvInit.map_config_func[tag]()
        engine = LLMEnvInit.getEngine(tag, **kwargs)
        logging.info(f"initLLMEngine: {tag}, {engine}")
        return engine

    @staticmethod
    def get_llm_llamacpp_args() -> dict:
        kwargs = {}
        kwargs["model_name"] = os.getenv("LLM_MODEL_NAME", "qwen2")
        if os.getenv("LLM_MODEL_PATH"):
            kwargs["model_path"] = os.getenv(
                "LLM_MODEL_PATH", os.path.join(MODELS_DIR, "qwen2-1_5b-instruct-q8_0.gguf")
            )
        elif os.getenv("LLM_MODEL_NAME_OR_PATH"):
            kwargs["model_path"] = os.getenv(
                "LLM_MODEL_NAME_OR_PATH", os.path.join(MODELS_DIR, "qwen2-1_5b-instruct-q8_0.gguf")
            )
        kwargs["model_type"] = os.getenv("LLM_MODEL_TYPE", "chat")
        kwargs["n_threads"] = os.cpu_count()
        kwargs["verbose"] = False
        kwargs["n_gpu_layers"] = int(os.getenv("N_GPU_LAYERS", "0"))
        kwargs["flash_attn"] = bool(os.getenv("FLASH_ATTN", ""))
        kwargs["llm_stream"] = True
        # if logger.getEffectiveLevel() != logging.DEBUG:
        #    kwargs["verbose"] = False
        kwargs["llm_stop"] = [
            "<|end|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<{/end}>",
            "</s>",
            "/s>",
            "</s",
            "<s>",
            "<|user|>",
            "<|assistant|>",
            "<|system|>",
        ]
        kwargs["llm_chat_system"] = os.getenv("LLM_CHAT_SYSTEM", DEFAULT_SYSTEM_PROMPT)
        kwargs["llm_tool_choice"] = os.getenv("LLM_TOOL_CHOICE", None)
        kwargs["chat_format"] = os.getenv("LLM_CHAT_FORMAT", None)
        kwargs["tokenizer_path"] = os.getenv("LLM_TOKENIZER_PATH", None)
        kwargs["clip_model_path"] = os.getenv("LLM_CLIP_MODEL_PATH", None)
        return kwargs

    @staticmethod
    def get_llm_llamacpp_generator_args() -> dict:
        kwargs = {}
        if os.getenv("LLM_MODEL_PATH"):
            kwargs["model_path"] = os.getenv(
                "LLM_MODEL_PATH", os.path.join(MODELS_DIR, "qwen2-1_5b-instruct-q8_0.gguf")
            )
        elif os.getenv("LLM_MODEL_NAME_OR_PATH"):
            kwargs["model_path"] = os.getenv(
                "LLM_MODEL_NAME_OR_PATH", os.path.join(MODELS_DIR, "qwen2-1_5b-instruct-q8_0.gguf")
            )
        kwargs["n_threads"] = os.getenv("N_THREADS", os.cpu_count())
        kwargs["verbose"] = bool(os.getenv("LLM_VERBOSE", ""))
        kwargs["n_gpu_layers"] = int(os.getenv("N_GPU_LAYERS", "0"))
        kwargs["flash_attn"] = bool(os.getenv("FLASH_ATTN", ""))
        # if logger.getEffectiveLevel() != logging.DEBUG:
        #    kwargs["verbose"] = False
        kwargs["tokenizer_path"] = os.getenv("LLM_TOKENIZER_PATH", None)
        return kwargs

    @staticmethod
    def get_llm_personal_ai_proxy_args() -> dict:
        kwargs = PersonalAIProxyArgs(
            api_url=os.getenv("API_URL", "http://localhost:8787/"),
            chat_bot=os.getenv("CHAT_BOT", "openai"),
            model_type=os.getenv("CHAT_TYPE", "chat_only"),
            openai_api_base_url=os.getenv("OPENAI_API_BASE_URL", "https://api.groq.com/openai/v1/"),
            model_name=os.getenv("LLM_MODEL_NAME", "llama3-70b-8192"),
            llm_chat_system=os.getenv("LLM_CHAT_SYSTEM", DEFAULT_SYSTEM_PROMPT),
            llm_stream=False,
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
            func_search_name=os.getenv("FUNC_SEARCH_NAME", "search_api"),
            func_weather_name=os.getenv("FUNC_WEATHER_NAME", "openweathermap_api"),
        ).__dict__
        return kwargs

    @staticmethod
    def _get_llm_generate_args() -> dict:
        from src.types.llm.sampling import LMGenerateArgs

        return LMGenerateArgs(
            lm_gen_seed=int(os.getenv("LLM_GEN_SEED", "42")),
            lm_gen_do_sample=bool(os.getenv("LLM_GEN_DO_SAMPLE", "1")),
            lm_gen_max_new_tokens=int(os.getenv("LLM_GEN_MAX_NEW_TOKENS", "1024")),
            lm_gen_temperature=float(os.getenv("LLM_GEN_TEMPERATURE", "0.8")),
            lm_gen_top_k=int(os.getenv("LLM_GEN_TOP_K", "50")),
            lm_gen_top_p=float(os.getenv("LLM_GEN_TOP_P", "0.95")),
            lm_gen_min_p=float(os.getenv("LLM_GEN_MIN_P", "0.0")),
            lm_gen_repetition_penalty=float(os.getenv("LLM_GEN_REPETITION_PENALTY", "1.1")),
            lm_gen_min_new_tokens=int(os.getenv("LLM_GEN_MIN_NEW_TOKENS", "0")),
        ).__dict__

    @staticmethod
    def _get_llm_transformers_args() -> dict:
        from src.types.llm.transformers import TransformersLMArgs

        kwargs = TransformersLMArgs(
            lm_model_name_or_path=os.getenv(
                "LLM_MODEL_NAME_OR_PATH", os.path.join(MODELS_DIR, "Qwen/Qwen2-0.5B-Instruct")
            ),
            lm_attn_impl=os.getenv("LLM_ATTN_IMPL", None),
            lm_device=os.getenv("LLM_DEVICE", None),
            lm_device_map=os.getenv("LLM_DEVICE_MAP", None),
            lm_torch_dtype=os.getenv("LLM_TORCH_DTYPE", "auto"),
            lm_stream=bool(os.getenv("LLM_STREAM", "1")),
            init_chat_prompt=os.getenv("LLM_INIT_CHAT_PROMPT", ""),
            chat_history_size=int(os.getenv("LLM_CHAT_HISTORY_SIZE", "10")),  # cache 10 round
            model_type=os.getenv("LLM_MODEL_TYPE", "chat_completion"),
            warmup_steps=int(os.getenv("LLM_WARMUP_STEPS", "1")),
            **LLMEnvInit._get_llm_generate_args(),
        ).__dict__
        return kwargs

    @staticmethod
    def get_llm_transformers_args() -> dict:
        hf_parser_file = os.getenv("HF_TRANSFOMERS_PARSER_FILE", None)
        if hf_parser_file:
            from .transformers import TransformersLLMInit

            return TransformersLLMInit.get_transformers_llm_args(hf_parser_file)
        return LLMEnvInit._get_llm_transformers_args()

    @staticmethod
    def get_llm_transformers_manual_image_janus_flow_args() -> dict:
        kwargs = LLMEnvInit.get_llm_transformers_args()
        kwargs["vae_model_name_or_path"] = os.getenv(
            "VAE_MODEL_NAME_OR_PATH",
            os.path.join(MODELS_DIR, "stabilityai/sdxl-vae"),
        )
        return kwargs

    @staticmethod
    def get_llm_vllm_generator_args() -> dict:
        # from src.types.llm.vllm import VllmEngineArgs, AsyncEngineArgs

        kwargs = dict(
            serv_args=dict(
                model=os.getenv(
                    "LLM_MODEL_NAME_OR_PATH", os.path.join(MODELS_DIR, "Qwen/Qwen2.5-0.5B")
                ),
                dtype=os.getenv("LLM_TORCH_DTYPE", "auto"),
                kv_cache_dtype=os.getenv("LLM_KV_CACHE_DTYPE", "auto"),
                gpu_memory_utilization=float(
                    os.getenv(
                        # The fraction of GPU memory to be used for the model executor, which can range from 0 to 1. For example, a value of 0.5 would imply 50% GPU memory utilization. If unspecified, will use the default value of 0.9. This is a per-instance limit, and only applies to the current vLLM instance.It does not matter if you have another vLLM instance running on the same GPU. For example, if you have two vLLM instances running on the same GPU, you can set the GPU memory utilization to 0.5 for each instance.
                        "LLM_GPU_MEMORY_UTILIZATION",
                        "0.9",  # default 0.9
                    )
                ),  # diff verl(rlfh inference)
            ),
            gen_args=LLMEnvInit._get_llm_generate_args(),
        )
        return kwargs

    @staticmethod
    def get_llm_sglang_generator_args() -> dict:
        # from src.types.llm.sglang import SGLangEngineArgs, ServerArgs

        kwargs = dict(
            serv_args=dict(
                model_path=os.getenv(
                    "LLM_MODEL_NAME_OR_PATH", os.path.join(MODELS_DIR, "Qwen/Qwen2.5-0.5B")
                ),
                dtype=os.getenv("LLM_TORCH_DTYPE", "auto"),
                kv_cache_dtype=os.getenv("LLM_KV_CACHE_DTYPE", "auto"),
                mem_fraction_static=float(
                    os.getenv(
                        # Fraction of the free GPU memory used for static memory like model weights and KV cache. If building KV cache fails, it should be increased. If CUDA runs out of memory, it should be decreased.
                        "LLM_GPU_MEMORY_UTILIZATION",
                        "0.6",  # default 0.88
                    )
                ),  # from vllm defined gpu_memory_utilization, diff verl(rlfh inference)
            ),
            gen_args=LLMEnvInit._get_llm_generate_args(),
        )
        return kwargs

    @staticmethod
    def get_llm_trtllm_generator_args() -> dict:
        # from src.types.llm.tensorrt_llm import TensorRTLLMEngineArgs, LlmArgs

        kwargs = dict(
            serv_args=dict(
                model=os.getenv(
                    "LLM_MODEL_NAME_OR_PATH", os.path.join(MODELS_DIR, "Qwen/Qwen2.5-0.5B")
                ),
                dtype=os.getenv("LLM_TORCH_DTYPE", "auto"),
                kv_cache_config=dict(
                    # The fraction of GPU memory fraction that should be allocated for the KV cache. Default is 90%. If both `max_tokens` and `free_gpu_memory_fraction` are specified, memory corresponding to the minimum will be used.
                    free_gpu_memory_fraction=float(
                        os.getenv("LLM_GPU_MEMORY_UTILIZATION", "0.7")
                    ),  # default 0.9
                ),
            ),
            gen_args=LLMEnvInit._get_llm_generate_args(),
        )
        return kwargs

    @staticmethod
    def get_llm_trtllm_runner_generator_args() -> dict:
        # from src.types.llm.tensorrt_llm import TensorRTLLMRunnerEngineArgs, TensorRTLLMRunnerArgs

        kwargs = dict(
            serv_args=dict(
                engine_dir=os.getenv(
                    "LLM_MODEL_NAME_OR_PATH", os.path.join(MODELS_DIR, "Qwen/Qwen2.5-0.5B-trtllm")
                ),  # build dtype engine
                debug_mode=bool(os.getenv("LLM_DEBUG_MODE", "")),
            ),
            gen_args=LLMEnvInit._get_llm_generate_args(),
        )
        return kwargs

    # TAG : config
    map_config_func = {
        "llm_llamacpp": get_llm_llamacpp_args,
        "llm_personalai_proxy": get_llm_personal_ai_proxy_args,
        "llm_transformers_manual": get_llm_transformers_args,
        "llm_transformers_pipeline": get_llm_transformers_args,
        "llm_transformers_manual_vision_qwen": get_llm_transformers_args,
        "llm_transformers_manual_vision_llama": get_llm_transformers_args,
        "llm_transformers_manual_vision_molmo": get_llm_transformers_args,
        "llm_transformers_manual_vision_deepseek": get_llm_transformers_args,
        "llm_transformers_manual_vision_janus": get_llm_transformers_args,
        "llm_transformers_manual_image_janus": get_llm_transformers_args,
        "llm_transformers_manual_vision_janus_flow": get_llm_transformers_args,
        "llm_transformers_manual_image_janus_flow": get_llm_transformers_manual_image_janus_flow_args,
        "llm_transformers_manual_vision_minicpmo": get_llm_transformers_args,
        "llm_transformers_generator": get_llm_transformers_args,
        "llm_llamacpp_generator": get_llm_llamacpp_generator_args,
        "llm_vllm_generator": get_llm_vllm_generator_args,
        "llm_sglang_generator": get_llm_sglang_generator_args,
        "llm_trtllm_generator": get_llm_trtllm_generator_args,
        "llm_trtllm_runner_generator": get_llm_trtllm_runner_generator_args,
    }
