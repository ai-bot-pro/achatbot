import mcp.types as types

from src.common.register import Register

prompts = Register("mcp-prompts")


def prompt_list() -> list[types.Prompt]:
    return [
        types.Prompt(
            name="simple",
            description="A simple prompt that can take optional context and topic arguments",
            arguments=[
                types.PromptArgument(
                    name="context",
                    description="Additional context to consider",
                    required=False,
                ),
                types.PromptArgument(
                    name="topic",
                    description="Specific topic to focus on",
                    required=False,
                ),
            ],
        )
    ]
