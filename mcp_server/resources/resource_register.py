import mcp.types as types
from pydantic import AnyUrl

from src.common.register import Register

resources = Register("mcp-resources")


SAMPLE_RESOURCES = {
    "greeting": "Hello! This is a sample text resource.",
    "help": "This server provides a few sample text resources for testing.",
    "about": "This is the simple-resource MCP server implementation.",
}


# https://modelcontextprotocol.io/specification/2025-03-26/server/resources
def resource_list() -> list[types.Prompt]:
    return [
        types.Resource(
            uri=AnyUrl(f"file:///{name}.txt"),
            name=name,
            description=f"A sample text resource named {name}",
            mimeType="text/plain",
        )
        for name in SAMPLE_RESOURCES.keys()
    ]
