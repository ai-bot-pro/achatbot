import os
import logging

import mcp.types as types

if os.getenv("SERVER_MODE", None):
    from mcp.server.fastmcp import FastMCP

    app = FastMCP("mcp-tools")
else:
    from mcp.server.lowlevel import Server

    app = Server("mcp-tools")


logger = logging.getLogger(__name__)


@app.list_tools()
async def list_tools() -> list[types.Tool]:
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
