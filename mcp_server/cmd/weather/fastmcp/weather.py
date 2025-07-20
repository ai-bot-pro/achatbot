"""
FastMCP Echo Server
"""

from fastmcp import Context
from mcp.types import (
    ToolAnnotations,
)

from achatbot.modules.functions.weather.api import WeatherFuncEnvInit
from achatbot.common.session import Session

from . import mcp

# https://gofastmcp.com/servers/tools


# NOTE: dispatch to execute
item = WeatherFuncEnvInit.initWeatherEngine().get_tool_call()
function = item.get("function")
if function is None:

    @mcp.tool(name="get weather", description="get weather")
    def get_weather(longitude: float, latitude: float, ctx: Context) -> str:
        session = Session(client_id=ctx.client_id)
        return WeatherFuncEnvInit.initWeatherEngine().execute(session, longitude, latitude)
else:

    @mcp.tool(
        name=function.get("name"),
        description=function.get("description"),
        annotations=ToolAnnotations(title="weather"),
    )
    def get_weather(longitude: float, latitude: float, ctx: Context) -> str:
        session = Session(client_id=ctx.client_id)
        return WeatherFuncEnvInit.initWeatherEngine().execute(
            session, **{"longitude": longitude, "latitude": latitude}
        )
