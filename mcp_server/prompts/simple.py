from .. import app, types


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


@app.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    return [
        types.Prompt(
            name="simple",
            description="A simple prompt that can take optional context and topic " "arguments",
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


@app.get_prompt()
async def get_prompt(name: str, arguments: dict[str, str] | None = None) -> types.GetPromptResult:
    if name != "simple":
        raise ValueError(f"Unknown prompt: {name}")

    if arguments is None:
        arguments = {}

    return types.GetPromptResult(
        messages=create_messages(context=arguments.get("context"), topic=arguments.get("topic")),
        description="A simple prompt with optional context and topic arguments",
    )
