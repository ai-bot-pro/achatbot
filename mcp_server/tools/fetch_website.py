from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.server.lowlevel import Server
import mcp.types as types

from .tool_register import functions


@functions.register("fetch_website")
async def fetch_website(
    app: Server,
    arguments: dict,
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    url = arguments.get("url")
    headers = {"User-Agent": "MCP Test Server (github.com/modelcontextprotocol/python-sdk)"}
    async with create_mcp_http_client(headers=headers) as client:
        response = await client.get(url)
        response.raise_for_status()
        return [types.TextContent(type="text", text=response.text)]
