
# init
use adk to develop agent
```shell
pip install google-adk
cd a2a_server/cmd
adk create my_agent
```

# dev
```shell
# local terminal
adk run my_agent

# ui
adk web --port 8000
```

# run agent server
- agent don't use ai framework, just use a2a adk and llm `chat/completions` api
```shell
# run a2a agent server port :6666
python -m a2a_server.cmd.mcp_github
```

# run client
- agent don't use ai framework, just use a2a adk, llm `chat/completions` api and use `A2AClient` or ClientFactory create a `Client` to `send_message`
```shell
python -m a2a_server.cmd.client --question "what's achatbot?"
```

---

function call(tool) -> mcp -> agent (mcp/tool + LLM) -> A2A -> personal Doraemon

have fun :)