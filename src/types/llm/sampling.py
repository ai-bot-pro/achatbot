from dataclasses import dataclass, field


@dataclass
class LMGenerateArgs:
    """lm generation sampling arguments"""

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
        default=20,
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
    # https://github.com/huggingface/transformers/issues/27670
    lm_gen_min_p: float = field(
        default=0.0,
        metadata={
            "help": "samples from tokens with probability larger than min_p * highest_token_probability. Default is 0.0."
        },
    )
    lm_gen_repetition_penalty: float = field(
        default=1.0,
        metadata={
            "help": "Controls the token repetition pealty.  no repetition Default is 1.0, >1.0: low repetition, <1.0: high repetition"
        },
    )
