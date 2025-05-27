from mcp.shared._httpx_utils import create_mcp_http_client

from .. import app, types


@app.call_tool()
async def call_tool(
    name: str, arguments: dict
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    if name != "fetch":
        raise ValueError(f"Unknown tool: {name}")
    if "url" not in arguments:
        raise ValueError("Missing required argument 'url'")
    return await fetch_website(arguments["url"])


async def fetch_website(
    url: str,
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    headers = {"User-Agent": "MCP Test Server (github.com/modelcontextprotocol/python-sdk)"}
    async with create_mcp_http_client(headers=headers) as client:
        response = await client.get(url)
        response.raise_for_status()
        return [types.TextContent(type="text", text=response.text)]
