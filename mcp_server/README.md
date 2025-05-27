# ❀❀ mcp ❀❀
KISS :)

# transports
- https://modelcontextprotocol.io/specification/2025-03-26/basic/transports

- stdio: use MCP for function with std in/out stream
- streamable-http: use MCP for function with http stream
- custom: use MCP for function with custom rpc stream https://modelcontextprotocol.io/specification/2025-03-26/basic/transports#custom-transports

# python sdk
- https://github.com/modelcontextprotocol/python-sdk/blob/main/README.md


> [!TIP]
> if no list_tools, use call_tool function doc string as tool meta info
> like serverless function, like cgi (u can use shell do this with mcp)
> u can public this tool pkg for others to use
> if use remote mcp server, need note [**security-warning**](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports#security-warning)
> use python magic method to do, keep it simple