from dataclasses import dataclass, field


@dataclass
class TransformersLMArgs:
    r"""
    HF transformers language model args (tiny,small,large,huge)
    https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig
    """

    lm_model_name_or_path: str = field(
        default="HuggingFaceTB/SmolLM-360M-Instruct",
        metadata={
            "help": "The pretrained language model to use. Default is 'HuggingFaceTB/SmolLM-360M-Instruct'."
        },
    )
    lm_device_map: str | dict | None = field(
        default=None,
        metadata={
            "help": "The device map for multi cpu/gpu. use dict for GPU acceleration, 'mps','cpu','auto', dict, None. default None"
        },
    )
    lm_device: str = field(
        default="cpu",
        metadata={
            "help": "The device for single cpu/mps/gpu. use 'cuda' for GPU acceleration, 'mps'(apple), 'cpu'. default cuda"
        },
    )
    lm_torch_dtype: str = field(
        default="auto",
        metadata={
            "help": "The PyTorch data type for the model and input tensors. One of `float32` (full-precision), `float16` or `bfloat16` (both half-precision), auto. default auto."
        },
    )
    # https://huggingface.co/docs/transformers/perf_infer_gpu_one
    lm_attn_impl: str | None = field(
        default=None,
        metadata={
            "help": "The attention implementation to use. One of 'sdpa', 'flash_attention_2', default no attention implementation."
        },
    )
    user_role: str = field(
        default="user",
        metadata={"help": "Role assigned to the user in the chat context. Default is 'user'."},
    )
    warnup_prompt: str = field(
        default="Repeat the word 'weedge niu bi'.",
        metadata={"help": "warnup llm generate prompt. Default is 'weedge niu bi'."},
    )
    warmup_steps: int = field(
        default=2,
        metadata={"help": "The number of steps to run the warmup prompt. Default is 2."},
    )
    init_chat_role: str = field(
        default="system",
        metadata={"help": "Initial role for setting up the chat context. Default is 'system'."},
    )
    init_chat_prompt: str = field(
        default="You are a helpful and friendly AI assistant. You are polite, respectful, and aim to provide concise responses of less than 20 words.",
        metadata={
            "help": "The initial chat prompt to establish context for the language model. Default is 'You are a helpful AI assistant.'"
        },
    )
    lm_gen_seed: int = field(
        default=42,
        metadata={"help": "The seed to use for the language model. Default is 42."},
    )
    lm_max_length: int = field(
        default=2048,
        metadata={
            "help": "Corresponds to the length of the input prompt + max_new_tokens. Its effect is overridden by max_new_tokens, if also set. Default is 2048."
        },
    )
    lm_gen_max_new_tokens: int = field(
        default=1024,
        metadata={
            "help": "Maximum number of new tokens to generate in a single completion. Default is 1024."
        },
    )
    lm_gen_min_new_tokens: int = field(
        default=0,
        metadata={
            "help": "Minimum number of new tokens to generate in a single completion. Default is 0."
        },
    )
    lm_gen_do_sample: bool = field(
        default=True,
        metadata={
            "help": "Whether to use sampling; set this to False for deterministic outputs. Default is False."
        },
    )
    lm_gen_temperature: float = field(
        default=0.1,
        metadata={
            "help": "Controls the randomness of the output. Set to 0.0 for deterministic (repeatable) outputs. Default is 0.1."
        },
    )
    lm_gen_top_k: int = field(
        default=1,
        metadata={
            "help": "Changing the top - k parameter sets the size of the shortlist the model samples from as it outputs each token. Setting top - k to 1 gives us greedy decoding. Default is 1"
        },
    )
    lm_gen_top_p: float = field(
        default=0.8,
        metadata={
            "help": "Top-p is usually set to a high value (like 0.75) with the purpose of limiting the long tail of low-probability tokens that may be sampled. We can use both top-k and top-p together. If both k and p are enabled, p acts after k. Default is 0.8."
        },
    )
    lm_gen_repetition_penalty: float = field(
        default=1.0,
        metadata={
            "help": "Controls the token repetition pealty.  no repetition Default is 1.0, >1.0: low repetition, <1.0: high repetition"
        },
    )
    lm_tokenizer_decode_batch_size: int = field(
        default=60,
        metadata={"help": "Number of tokens to batch decode at a time"},
    )
    chat_history_size: int | None = field(
        default=None,
        metadata={
            "help": "Number of interactions assitant-user to keep for the chat. None for no limitations, <= 0 for no history"
        },
    )
    lm_stream: bool = field(
        default=True,
        metadata={
            "help": "Whether to use streaming; set this to True for streaming output. Default is True."
        },
    )
    model_type: str = field(
        default="chat_completion",
        metadata={
            "help": "Model type, generate, chat_completion(chat/instruct). Default is chat_completion"
        },
    )
    lm_bnb_quant_type: str = field(
        default="int4",
        metadata={"help": "The BitsAndBytes quantization type, default int4."},
    )

    def to_dict(self) -> dict:
        return {
            field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()
        }


@dataclass
class TransformersSpeechLMArgs(TransformersLMArgs):
    """
    text+speech lm args
    """

    def to_dict(self) -> dict:
        return super().to_dict()


@dataclass
class TransformersImageLMArgs(TransformersLMArgs):
    """
    text+Image lm args
    """

    vae_model_name_or_path: str = field(
        default="stabilityai/sdxl-vae",
        metadata={"help": "The diffusion model to use. Default is 'stabilityai/sdxl-vae'."},
    )

    def to_dict(self) -> dict:
        return super().to_dict()
