# ❀❀ weather mcp server ❀❀
use fast mcp server for python, KISS

# env
```shell
# use openweathermap_api
FUNC_WEATHER_TAG=openweathermap_api
OPEN_WEATHER_MAP_UNITS=metric
OPEN_WEATHER_MAP_LANG=zh_cn
# https://home.openweathermap.org/api_keys
OPENWEATHERMAP_API_KEY=XXX
```

# transports
- https://modelcontextprotocol.io/specification/2025-03-26/basic/transports

- stdio: use MCP for function with std in/out stream
- streamable-http: use MCP for function with http stream
- custom: use MCP for function with custom rpc stream https://modelcontextprotocol.io/specification/2025-03-26/basic/transports#custom-transports

# ⭐️ fastcmp sdk ⭐️ (nice doc)
- https://github.com/jlowin/fastmcp
- https://gofastmcp.com/getting-started/welcome


# debug develop
- run fast mcp server
```shell
# help
python -m mcp_server.cmd.weather.fastmcp --help

# defualt transport is stdio
python -m mcp_server.cmd.weather.fastmcp
# sse
python -m mcp_server.cmd.weather.fastmcp --transport sse
# stateless-streamable-http
python -m mcp_server.cmd.weather.fastmcp --transport streamable-http --stateless-http
# state-streamable-http
python -m mcp_server.cmd.weather.fastmcp --transport streamable-http
```
- The MCP Inspector is an interactive developer tool for testing and debugging MCP servers
```shell
# NOTE: inspector now just support stdio transport
# https://modelcontextprotocol.io/docs/tools/inspector
# run inspector page to debug  mcp server
npx @modelcontextprotocol/inspector
# or run with mcp stdio transport server
npx @modelcontextprotocol/inspector `which python` -m mcp_server.cmd.weather.fastmcp
```