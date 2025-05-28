import logging

import click

from . import mcp


@click.command()
@click.option(
    "--port",
    default=8000,
    help="Port to listen on for SSE or state-streamable-http, stateless-streamable-http",
)
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse", "state-streamable-http", "stateless-streamable-http"]),
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
def main(
    port: int,
    transport: str,
    log_level: str,
    json_response: bool,
) -> int:
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    mcp.run()
