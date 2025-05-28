import logging

import click

from . import mcp
from .weather import *


@click.command()
@click.option(
    "--host",
    default="127.0.0.1",
    help="Host to listen on for SSE or streamable-http",
)
@click.option(
    "--port",
    default=8000,
    help="Port to listen on for SSE or streamable-http",
)
@click.option(
    "--transport",
    type=click.Choice(
        [
            "stdio",
            "sse",
            "streamable-http",
        ]
    ),
    default="stdio",
    help="Transport type",
)
@click.option(
    "--log-level",
    default="INFO",
    help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
)
@click.option(
    "--json-response",
    is_flag=True,
    default=False,
    help="Enable JSON responses instead of SSE streams",
)
@click.option(
    "--stateless-http",
    is_flag=True,
    default=False,
    help="Enable stateless HTTP transport",
)
def main(
    host: str,
    port: int,
    transport: str,
    log_level: str,
    json_response: bool,
    stateless_http: bool,
) -> int:
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    mcp.settings.json_response = json_response
    mcp.settings.stateless_http = stateless_http
    logging.info(f"Starting MCP Server, {mcp.settings=}")

    if transport == "streamable-http":
        mcp.run(
            transport=transport,
            host=host,
            port=port,
            path="/mcp",
            log_level=log_level.lower(),
        )
    elif transport == "sse":
        mcp.run(
            transport=transport,
            host=host,
            port=port,
            log_level=log_level.lower(),
        )
    else:
        mcp.run(transport=transport)
