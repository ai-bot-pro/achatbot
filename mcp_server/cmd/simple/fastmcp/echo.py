"""
FastMCP Echo Server
"""

from fastmcp import Context

from . import mcp

# https://gofastmcp.com/servers/tools


@mcp.tool()
async def echo_context(text: str, ctx: Context):
    # Log a message to the client
    await ctx.info(f"Processing {text}...")

    # Ask client LLM to summarize the data
    # sampling to mock
    summary = await ctx.sample(f"Summarize: {text}")

    # Return the summary
    return summary.text


@mcp.tool()
def echo_tool(text: str) -> str:
    """Echo the input text"""
    return text


@mcp.resource("echo://static")
def echo_resource() -> str:
    return "Echo!"


@mcp.resource("echo://{text}")
def echo_template(text: str) -> str:
    """Echo the input text"""
    return f"Echo: {text}"


@mcp.prompt("echo")
def echo_prompt(text: str) -> str:
    return text
