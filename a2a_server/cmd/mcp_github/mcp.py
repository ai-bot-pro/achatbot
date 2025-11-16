import asyncio
from pathlib import Path

from jinja2 import Template
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.types import CallToolResult, TextContent

dir_path = Path(__file__).parent

with Path(dir_path / "tpl" / "tool.jinja").open("r") as f:
    template = Template(f.read())


async def get_mcp_tool_prompt(url: str) -> str:
    """Get the MCP tool prompt for a given URL.

    Args:
        url (str): The URL of the MCP tool.

    Returns:
        str: The MCP tool prompt.
    """
    async with (
        sse_client(url) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()

        resources = await session.list_tools()
        return template.render(tools=resources.tools)


async def call_mcp_tool(url: str, tool_name: str, arguments: dict | None = None) -> CallToolResult:
    """Call an MCP tool with the given URL and tool name.

    Args:
        url (str): The URL of the MCP tool.
        tool_name (str): The name of the tool to call.
        arguments (dict | None, optional): The arguments to pass to the tool. Defaults to None.

    Returns:
        CallToolResult: The result of the tool call.
    """
    async with (
        sse_client(
            url=url,
        ) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()

        return await session.call_tool(tool_name, arguments=arguments)


"""
python -m a2a_server.cmd.mcp_github.mcp
"""
if __name__ == "__main__":
    print(asyncio.run(get_mcp_tool_prompt("https://gitmcp.io/ai-bot-pro/achatbot")))
    result = asyncio.run(
        call_mcp_tool(
            "https://gitmcp.io/ai-bot-pro/achatbot",
            "fetch_achatbot_documentation",
        )
    )
    for content in result.content:
        if isinstance(content, TextContent):
            print(content.text)
