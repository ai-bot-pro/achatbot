from dataclasses import dataclass, field

from vllm import AsyncEngineArgs

from .sampling import LMGenerateArgs


@dataclass
class VllmEngineArgs:
    """
    vllm language model engine args
    """

    # https://docs.vllm.ai/en/latest/api/vllm/engine/arg_utils.html#vllm.engine.arg_utils.AsyncEngineArgs
    serv_args: dict = field(default_factory=lambda: AsyncEngineArgs().__dict__)
    gen_args: dict = field(default_factory=lambda: LMGenerateArgs().__dict__)

    user_role: str = field(
        default="user",
        metadata={"help": "Role assigned to the user in the chat context. Default is 'user'."},
    )
    assistant_role: str = field(
        default="assistant",
        metadata={
            "help": "Role assigned to the assistant in the chat context. Default is 'assistant'."
        },
    )
    warmup_prompt: str = field(
        default="Repeat the word 'weedge niu bi'.",
        metadata={"help": "warmup llm generate prompt. Default is 'weedge niu bi'."},
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
        default="",
        metadata={
            "help": "The initial chat prompt to establish context for the language model. Default is 'You are a helpful AI assistant.'"
        },
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
