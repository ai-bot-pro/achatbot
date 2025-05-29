import logging
import contextlib
from collections.abc import AsyncIterator

import click
import anyio
from pydantic import AnyUrl
from mcp.server.lowlevel import Server
import mcp.types as types

from mcp_server.resources.resource_register import resources, resource_list
from mcp_server.prompts.prompt_register import prompts, prompt_list
from achatbot.modules.functions.weather.api import WeatherFuncEnvInit
from achatbot.common.session import Session

app = Server(__name__)


@app.list_resources()
async def list_resources() -> list[types.Resource]:
    return resource_list()


@app.read_resource()
async def read_resource(uri: AnyUrl) -> str | bytes:
    logging.info(f"{resources.dict()=}")
    if uri.path not in resources.keys():
        raise ValueError(f"Unknown resource: {uri.path}")
    return await resources[uri.path](uri)


@app.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    return prompt_list()


@app.get_prompt()
async def get_prompt(
    name: str,
    arguments: dict[str, str] | None = None,
) -> types.GetPromptResult:
    logging.info(f"{prompts.dict()=}")
    if name not in prompts.keys():
        raise ValueError(f"Unknown prompt: {name}")
    return await prompts[name](arguments)


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    tools = []
    item = WeatherFuncEnvInit.initWeatherEngine().get_tool_call()
    function = item.get("function")
    if function is None:
        return tools
    tool = types.Tool(
        name=function.get("name"),
        description=function.get("description"),
        inputSchema=function.get("parameters"),
        annotations=types.ToolAnnotations(title="weather"),
    )
    tools.append(tool)
    return tools


# dispatch
@app.call_tool()
async def call_tool(
    name: str, arguments: dict
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    item = WeatherFuncEnvInit.initWeatherEngine().get_tool_call()
    function = item.get("function")
    if function is None:
        raise ValueError(f"no function found for {name}")
    logging.info(f"{item=}")
    if name != function.get("name"):
        raise ValueError(f"Unknown tool: {name}")

    session = Session(client_id=app.request_context.request_id)
    result = WeatherFuncEnvInit.initWeatherEngine().execute(session, **arguments)
    return [types.TextContent(type="text", text=result)]


def run_stdio():
    # https://modelcontextprotocol.io/specification/2025-03-26/basic/transports#stdio
    from mcp.server.stdio import stdio_server

    async def arun():
        async with stdio_server() as streams:
            await app.run(streams[0], streams[1], app.create_initialization_options())

    anyio.run(arun)


def run_state_streamable_http(
    host: str = "127.0.0.1", port: int = 8000, json_response: bool = False
):
    import uvicorn
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.routing import Mount
    from starlette.types import Receive, Scope, Send

    from mcp_server.common.event_store import InMemoryEventStore

    # Create event store for resumability
    # The InMemoryEventStore enables resumability support for StreamableHTTP transport.
    # It stores SSE events with unique IDs, allowing clients to:
    #   1. Receive event IDs for each SSE message
    #   2. Resume streams by sending Last-Event-ID in GET requests
    #   3. Replay missed events after reconnection
    # Note: This in-memory implementation is for demonstration ONLY.
    # For production, use a persistent storage solution.
    event_store = InMemoryEventStore()

    # Create the session manager with our app and event store
    session_manager = StreamableHTTPSessionManager(
        app=app,
        event_store=event_store,  # Enable resumability
        json_response=json_response,
    )

    # ASGI handler for streamable HTTP connections
    async def handle_streamable_http(scope: Scope, receive: Receive, send: Send) -> None:
        await session_manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        """Context manager for managing session manager lifecycle."""
        async with session_manager.run():
            logging.info("Application started with StreamableHTTP session manager!")
            try:
                yield
            finally:
                logging.info("Application shutting down...")

    # Create an ASGI application using the transport
    starlette_app = Starlette(
        debug=True,
        routes=[
            Mount("/mcp", app=handle_streamable_http),
        ],
        lifespan=lifespan,
    )

    uvicorn.run(starlette_app, host=host, port=port)


def run_stateless_streamable_http(
    host: str = "127.0.0.1", port: int = 8000, json_response: bool = False
):
    import uvicorn
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.routing import Mount
    from starlette.types import Receive, Scope, Send

    # Create the session manager with true stateless mode
    session_manager = StreamableHTTPSessionManager(
        app=app,
        event_store=None,
        json_response=json_response,
        stateless=True,
    )

    async def handle_streamable_http(scope: Scope, receive: Receive, send: Send) -> None:
        await session_manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        """Context manager for session manager."""
        async with session_manager.run():
            logging.info("Application started with StreamableHTTP session manager!")
            try:
                yield
            finally:
                logging.info("Application shutting down...")

    # Create an ASGI application using the transport
    starlette_app = Starlette(
        debug=True,
        routes=[
            Mount("/mcp", app=handle_streamable_http),
        ],
        lifespan=lifespan,
    )

    uvicorn.run(starlette_app, host=host, port=port)


def run_sse(host: str, port: int):
    # https://modelcontextprotocol.io/specification/2025-03-26/basic/transports#streamable-http
    # https://modelcontextprotocol.io/specification/2025-03-26/basic/transports#backwards-compatibility
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.responses import Response
    from starlette.routing import Mount, Route
    import uvicorn

    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
            await app.run(streams[0], streams[1], app.create_initialization_options())
        return Response()

    starlette_app = Starlette(
        debug=True,
        routes=[
            Route("/sse", endpoint=handle_sse, methods=["GET"]),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

    uvicorn.run(starlette_app, host=host, port=port)


@click.command()
@click.option(
    "--host",
    default="127.0.0.1",
    help="Host to listen on for SSE or state-streamable-http, stateless-streamable-http",
)
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
    host: str,
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

    if transport == "sse":
        logging.info("run_sse")
        run_sse(host, port)
    elif transport == "state-streamable-http":
        logging.info("run_state_streamable_http")
        run_state_streamable_http(host=host, port=port, json_response=json_response)
    elif transport == "stateless-streamable-http":
        logging.info("run_stateless_streamable_http")
        run_stateless_streamable_http(host=host, port=port, json_response=json_response)
    else:
        logging.info("run_stdio")
        run_stdio()

    return 0
