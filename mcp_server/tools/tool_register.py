import mcp.types as types

from src.common.register import Register

functions = Register("mcp-tools")


def tool_list() -> list[types.Tool]:
    return [
        types.Tool(
            name="fetch_website",
            description="Fetches a website and returns its content",
            inputSchema={
                "type": "object",
                "required": ["url"],
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to fetch",
                    }
                },
            },
        ),
        types.Tool(
            name="stateless_send_notification",
            description=("Sends a stream of notifications with configurable count and interval"),
            inputSchema={
                "type": "object",
                "required": ["interval", "count", "caller"],
                "properties": {
                    "interval": {
                        "type": "number",
                        "description": "Interval between notifications in seconds",
                    },
                    "count": {
                        "type": "number",
                        "description": "Number of notifications to send",
                    },
                    "caller": {
                        "type": "string",
                        "description": ("Identifier of the caller to include in notifications"),
                    },
                },
            },
        ),
        types.Tool(
            name="state_send_notification",
            description=("Sends a stream of notifications with configurable count and interval"),
            inputSchema={
                "type": "object",
                "required": ["interval", "count", "caller"],
                "properties": {
                    "interval": {
                        "type": "number",
                        "description": "Interval between notifications in seconds",
                    },
                    "count": {
                        "type": "number",
                        "description": "Number of notifications to send",
                    },
                    "caller": {
                        "type": "string",
                        "description": ("Identifier of the caller to include in notifications"),
                    },
                },
            },
        ),
    ]
