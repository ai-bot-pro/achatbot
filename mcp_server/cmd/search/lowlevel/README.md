# ❀❀ search mcp server ❀❀
use Low-Level Server

# env
```shell
# use search_api 
FUNC_SEARCH_TAG=search_api
SEARCH_ENGINE=google
SEARCH_GL=cn
SEARCH_HL=zh-cn
SERPER_PAGE=1
SERPER_NUM=5
#https://www.searchapi.io/api_tokens
SEARCH_API_KEY=XXX

# use search1_api
FUNC_SEARCH_TAG=search1_api
SEARCH1_ENGINE=google
SEARCH1_IMAGE=
CRAWL_RESULTS=0
MAX_RESULTS=5
#https://dashboard.search1api.com/api-keys
SEARCH1_API_KEY=XXX

# use serper_api
FUNC_SEARCH_TAG=serper_api
SERPER_GL=cn
SERPER_HL=zh-cn
SERPER_PAGE=1
SERPER_NUM=5
#https://serper.dev/api-key
SERPER_API_KEY=XXX
```

# transports
- https://modelcontextprotocol.io/specification/2025-03-26/basic/transports

- stdio: use MCP for function with std in/out stream
- streamable-http: use MCP for function with http stream
- custom: use MCP for function with custom rpc stream https://modelcontextprotocol.io/specification/2025-03-26/basic/transports#custom-transports

# python sdk
- https://github.com/modelcontextprotocol/python-sdk/blob/main/README.md


> [!TIP]
> - if no list_tools, use call_tool function doc string as tool meta info
> - like serverless function, like cgi (u can use shell do this with mcp)
> - u can public this tool pkg for others to use
> - if use remote mcp server, need note [**security-warning**](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports#security-warning)
> use python magic method to do, keep it search
> - if mcp server is deployed in cloud for online service, u can use golang/rust to develop mcp server with docker for CI/CD, like [github-mcp-server](https://github.com/github/github-mcp-server)

# debug develop
- run mcp server
```shell
# help
python -m mcp_server.cmd.search.lowlevel --help

# defualt transport is stdio
python -m mcp_server.cmd.search.lowlevel
# sse
python -m mcp_server.cmd.search.lowlevel --transport sse
# stateless-streamable-http
python -m mcp_server.cmd.search.lowlevel --transport stateless-streamable-http
# state-streamable-http
python -m mcp_server.cmd.search.lowlevel --transport state-streamable-http
```
- The MCP Inspector is an interactive developer tool for testing and debugging MCP servers
```shell
# NOTE: inspector now just support stdio transport
# https://modelcontextprotocol.io/docs/tools/inspector
# run inspector page to debug  mcp server
npx @modelcontextprotocol/inspector
# or run with mcp stdio transport server 
npx @modelcontextprotocol/inspector `which python` -m mcp_server.cmd.search.lowlevel
```
> [!NOTE]
> - if mcp server use stdio transport, don't run multi inspector page, it will cause error
> - use Low-Level Server, call_tool function only use once. do dispatch with other function