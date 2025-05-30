"""
FastMCP Echo Server
"""

from fastmcp import Context
from mcp.types import (
    ToolAnnotations,
)

from achatbot.modules.functions.search.api import SearchFuncEnvInit
from achatbot.common.session import Session

from . import mcp

# https://gofastmcp.com/servers/tools


# NOTE: dispatch to execute
item = SearchFuncEnvInit.initSearchEngine().get_tool_call()
function = item.get("function")
if function is None:

    @mcp.tool(name="web_search", description="web search")
    def web_search(query: str, ctx: Context) -> str:
        session = Session(client_id=ctx.client_id)
        return SearchFuncEnvInit.initSearchEngine().execute(session, **{"query": query})
else:

    @mcp.tool(
        name=function.get("name"),
        description=function.get("description"),
        annotations=ToolAnnotations(title="search"),
    )
    def web_search(query: str, ctx: Context) -> str:
        session = Session(client_id=ctx.client_id)
        return SearchFuncEnvInit.initSearchEngine().execute(session, **{"query": query})
