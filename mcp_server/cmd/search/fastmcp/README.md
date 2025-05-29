# ❀❀ search mcp server ❀❀
use fast mcp server for python, KISS

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

# ⭐️ fastcmp sdk ⭐️ (nice doc)
- https://github.com/jlowin/fastmcp
- https://gofastmcp.com/getting-started/welcome


# debug develop
- run fast mcp server
```shell
# help
python -m mcp_server.cmd.search.fastmcp --help

# defualt transport is stdio
python -m mcp_server.cmd.search.fastmcp
# sse
python -m mcp_server.cmd.search.fastmcp --transport sse
# stateless-streamable-http
python -m mcp_server.cmd.search.fastmcp --transport streamable-http --stateless-http
# state-streamable-http
python -m mcp_server.cmd.search.fastmcp --transport streamable-http
```
- The MCP Inspector is an interactive developer tool for testing and debugging MCP servers
```shell
# NOTE: inspector now just support stdio transport
# https://modelcontextprotocol.io/docs/tools/inspector
# run inspector page to debug  mcp server
npx @modelcontextprotocol/inspector
# or run with mcp stdio transport server
npx @modelcontextprotocol/inspector `which python` -m mcp_server.cmd.search.fastmcp
```