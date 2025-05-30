import asyncio

import click
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client


async def search(session: ClientSession):
    # Initialize the connection
    await session.initialize()

    # List available resources
    resources = await session.list_resources()
    print(f"{resources=}")
    # Read a resource
    content, mime_type = await session.read_resource("file:///help.txt")
    print(f"{content=} {mime_type=}")

    # List available prompts
    prompts = await session.list_prompts()
    print(f"{prompts=}")

    # Get a prompt
    prompt = await session.get_prompt(
        "simple", arguments={"topic": "topic1", "context": "context1"}
    )
    print(f"{prompt=}")

    # List available tools
    tools = await session.list_tools()
    print(f"{tools=}")

    # Call a tool
    result = await session.call_tool("web_search", arguments={"query": "北京历史文物景点"})
    print(f"{result=}")


async def weather(session: ClientSession):
    # Initialize the connection
    await session.initialize()

    # List available resources
    resources = await session.list_resources()
    print(f"{resources=}")
    # Read a resource
    content, mime_type = await session.read_resource("file:///help.txt")
    print(f"{content=} {mime_type=}")

    # List available prompts
    prompts = await session.list_prompts()
    print(f"{prompts=}")

    # Get a prompt
    prompt = await session.get_prompt(
        "simple", arguments={"topic": "topic1", "context": "context1"}
    )
    print(f"{prompt=}")

    # List available tools
    tools = await session.list_tools()
    print(f"{tools=}")

    # Call a tool
    result = await session.call_tool("get_weather", arguments={"longitude": 1.1, "latitude": 2.2})
    print(f"{result=}")


async def simple(session: ClientSession):
    # Initialize the connection
    await session.initialize()

    # List available resources
    resources = await session.list_resources()
    print(f"{resources=}")
    # Read a resource
    content, mime_type = await session.read_resource("file:///help.txt")
    print(f"{content=} {mime_type=}")

    # List available prompts
    prompts = await session.list_prompts()
    print(f"{prompts=}")

    # Get a prompt
    prompt = await session.get_prompt(
        "simple", arguments={"topic": "topic1", "context": "context1"}
    )
    print(f"{prompt=}")

    # List available tools
    tools = await session.list_tools()
    print(f"{tools=}")

    # Call a tool
    result = await session.call_tool("fetch_website", arguments={"url": "https://www.baidu.com"})
    print(f"{result=}")


async def streamablehttp(func, name: str):
    # Connect to a streamable HTTP server
    async with streamablehttp_client("http://localhost:8000/mcp") as (
        read_stream,
        write_stream,
        _,
    ):
        # Create a session using the client streams
        async with ClientSession(read_stream, write_stream) as session:
            await func(session)


async def sse(func, name: str):
    # Connect to a sse HTTP server
    async with sse_client("http://localhost:8001/sse") as (read_stream, write_stream):
        # Create a session using the client streams
        async with ClientSession(read_stream, write_stream) as session:
            await func(session)


async def stdio(func, name: str):
    # Create server parameters for stdio connection
    server_params = StdioServerParameters(
        command="python",  # Executable
        args=["-m", f"mcp_server.cmd.{name}.lowlevel"],  # Optional command line arguments
        env=None,  # Optional environment variables
    )

    # Optional: create a sampling callback
    async def handle_sampling_message(
        message: types.CreateMessageRequestParams,
    ) -> types.CreateMessageResult:
        return types.CreateMessageResult(
            role="assistant",
            content=types.TextContent(
                type="text",
                text="Hello, world! from model",
            ),
            model="gpt-3.5-turbo",
            stopReason="endTurn",
        )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write, sampling_callback=handle_sampling_message) as session:
            await func(session)


@click.command()
@click.option(
    "--transport",
    default="stdio",
    type=click.Choice(["stdio", "sse", "streamable-http"]),
    help="transport",
)
@click.option(
    "--name",
    default="simple",
    help="mcp server name",
)
def main(transport: str, name: str) -> int:
    transport_funcs = {
        "stdio": stdio,
        "sse": sse,
        "streamable-http": streamablehttp,
    }
    server_funcs = {
        "simple": simple,
        "weather": weather,
        "search": search,
    }
    if server_funcs.get(name):
        asyncio.run(transport_funcs[transport](server_funcs[name], name))
    else:
        print(f"Unknown server name: {name}")
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
