import mcp.types as types

from .prompt_register import prompts


def create_messages(
    context: str | None = None, topic: str | None = None
) -> list[types.PromptMessage]:
    """Create the messages for the prompt."""
    messages = []

    # Add context if provided
    if context:
        messages.append(
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text", text=f"Here is some relevant context: {context}"
                ),
            )
        )

    # Add the main prompt
    prompt = "Please help me with "
    if topic:
        prompt += f"the following topic: {topic}"
    else:
        prompt += "whatever questions I may have."

    messages.append(
        types.PromptMessage(role="user", content=types.TextContent(type="text", text=prompt))
    )

    return messages


@prompts.register("simple")
async def get_prompt(
    arguments: dict[str, str] | None = None,
) -> types.GetPromptResult:
    if arguments is None:
        arguments = {}

    return types.GetPromptResult(
        messages=create_messages(context=arguments.get("context"), topic=arguments.get("topic")),
        description="A simple prompt with optional context and topic arguments",
    )
