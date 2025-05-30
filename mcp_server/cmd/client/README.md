# client

servers_config.json | define as agent tools config :)

> [!NOTE]
> if have same tool name, use last one

```json
{
  "mcpServers": {
    "simple": {
      "transport": "stdio",
      "parameters": {
        "command": "python",
        "args": ["-m", "mcp_server.cmd.simple.lowlevel"],
        "env": {}
      }
    },
    "weather": {
      "transport": "streamable-http",
      "parameters": {
        "url": "http://localhost:8000/mcp",
        "headers": {},
        "timeout": 30,
        "sse_read_timeout": 300,
        "terminate_on_close": true,
        "auth": null
      }
    },
    "search": {
      "transport": "sse",
      "parameters": {
        "url": "http://localhost:8001/sse",
        "headers": {},
        "timeout": 30,
        "sse_read_timeout": 300,
        "auth": null
      }
    }
  }
}
```

```shell
# run streamable-http mcp server weather
python -m mcp_server.cmd.weather.lowlevel --transport state-streamable-http
# run sse mcp server search
python -m mcp_server.cmd.search.lowlevel --transport sse --port 8001

# local terminal chat with stdio,sse,streamable-http transport mcp servers_config.json
python -m mcp_server.cmd.client.terminal
```

case: execute command and log in terminal
```
python -m mcp_server.cmd.client.terminal
2025-05-30 15:57:42,812 - root - INFO - run_stdio
2025-05-30 15:57:42,823 - INFO - Connecting to StreamableHTTP endpoint: http://localhost:8000/mcp
2025-05-30 15:57:42,933 - INFO - HTTP Request: POST http://localhost:8000/mcp "HTTP/1.1 307 Temporary Redirect"
2025-05-30 15:57:42,938 - INFO - HTTP Request: POST http://localhost:8000/mcp/ "HTTP/1.1 200 OK"
2025-05-30 15:57:42,938 - INFO - Received session ID: e8143c23506e4d4db07e72b42f90761f
2025-05-30 15:57:42,940 - INFO - Connecting to SSE endpoint: http://localhost:8001/sse
2025-05-30 15:57:42,963 - INFO - HTTP Request: POST http://localhost:8000/mcp "HTTP/1.1 307 Temporary Redirect"
2025-05-30 15:57:42,965 - INFO - HTTP Request: GET http://localhost:8000/mcp "HTTP/1.1 307 Temporary Redirect"
2025-05-30 15:57:42,966 - INFO - HTTP Request: GET http://localhost:8001/sse "HTTP/1.1 200 OK"
2025-05-30 15:57:42,968 - INFO - Received endpoint URL: http://localhost:8001/messages/?session_id=2fcb20d7741e489d860dbafb54803c0b
2025-05-30 15:57:42,968 - INFO - Starting post writer with endpoint URL: http://localhost:8001/messages/?session_id=2fcb20d7741e489d860dbafb54803c0b
2025-05-30 15:57:42,969 - INFO - HTTP Request: POST http://localhost:8000/mcp/ "HTTP/1.1 202 Accepted"
2025-05-30 15:57:42,971 - INFO - HTTP Request: GET http://localhost:8000/mcp/ "HTTP/1.1 200 OK"
2025-05-30 15:57:42,975 - INFO - HTTP Request: POST http://localhost:8001/messages/?session_id=2fcb20d7741e489d860dbafb54803c0b "HTTP/1.1 202 Accepted"
2025-05-30 15:57:42,978 - mcp.server.lowlevel.server - INFO - Processing request of type ListToolsRequest
2025-05-30 15:57:42,979 - INFO - HTTP Request: POST http://localhost:8001/messages/?session_id=2fcb20d7741e489d860dbafb54803c0b "HTTP/1.1 202 Accepted"
2025-05-30 15:57:42,985 - INFO - HTTP Request: POST http://localhost:8000/mcp "HTTP/1.1 307 Temporary Redirect"
2025-05-30 15:57:42,991 - INFO - HTTP Request: POST http://localhost:8000/mcp/ "HTTP/1.1 200 OK"
2025-05-30 15:57:42,996 - INFO - HTTP Request: POST http://localhost:8001/messages/?session_id=2fcb20d7741e489d860dbafb54803c0b "HTTP/1.1 202 Accepted"
You: 你好 
2025-05-30 15:58:05,455 - INFO - HTTP Request: POST https://api.together.xyz/v1/chat/completions "HTTP/1.1 200 OK"
2025-05-30 15:58:05,462 - INFO - 
Assistant: 你好！需要我帮你做些什么吗？
You: 查询下北京的经纬度
2025-05-30 15:58:22,536 - INFO - HTTP Request: POST https://api.together.xyz/v1/chat/completions "HTTP/1.1 200 OK"
2025-05-30 15:58:22,545 - INFO - 
Assistant: 北京的经纬度大约是经度116.4057，纬度39.9042。如果你需要更精确的坐标，请告诉我具体地点，我会帮你查询更准确的经纬度。
You: 查下北京今天的天气
2025-05-30 15:58:48,067 - INFO - HTTP Request: POST https://api.together.xyz/v1/chat/completions "HTTP/1.1 200 OK"
2025-05-30 15:58:48,073 - INFO - 
Assistant: {
    "tool": "get_weather",
    "arguments": {
        "longitude": "116.4057",
        "latitude": "39.9042"
    }
}
2025-05-30 15:58:48,073 - INFO - Executing tool: get_weather
2025-05-30 15:58:48,074 - INFO - With arguments: {'longitude': '116.4057', 'latitude': '39.9042'}
2025-05-30 15:58:48,077 - mcp.server.lowlevel.server - INFO - Processing request of type ListToolsRequest
2025-05-30 15:58:48,094 - INFO - HTTP Request: POST http://localhost:8000/mcp "HTTP/1.1 307 Temporary Redirect"
2025-05-30 15:58:48,097 - INFO - HTTP Request: POST http://localhost:8000/mcp/ "HTTP/1.1 200 OK"
2025-05-30 15:58:48,099 - INFO - Executing get_weather...
2025-05-30 15:58:48,103 - INFO - HTTP Request: POST http://localhost:8000/mcp "HTTP/1.1 307 Temporary Redirect"
2025-05-30 15:58:48,106 - INFO - HTTP Request: POST http://localhost:8000/mcp/ "HTTP/1.1 200 OK"
2025-05-30 15:58:50,795 - INFO - HTTP Request: POST https://api.together.xyz/v1/chat/completions "HTTP/1.1 200 OK"
2025-05-30 15:58:50,800 - INFO - 
Final response: 今天北京的天气状况如下：天空被厚厚的云层覆盖，有些阴沉。当前温度为28.93°C，体感温度为27.66°C，湿度为27%。风速为5.53米/秒，来自东南方向。天空的云量达到100%。白天的能见度良好，约为10000米。太阳升起的时间是早上6:12，落山时间是晚上7:15。整体来说，今天北京的天气以云层为主，温度适中，但请注意保持适当的衣物以应对体感温度。
You: 请帮吗搜索北京的著名景点有哪些?
2025-05-30 16:00:49,377 - INFO - HTTP Request: POST https://api.together.xyz/v1/chat/completions "HTTP/1.1 200 OK"
2025-05-30 16:00:49,391 - INFO - 
Assistant: {
    "tool": "web_search",
    "arguments": {
        "query": "北京著名景点"
    }
}
2025-05-30 16:00:49,391 - INFO - Executing tool: web_search
2025-05-30 16:00:49,392 - INFO - With arguments: {'query': '北京著名景点'}
2025-05-30 16:00:49,395 - mcp.server.lowlevel.server - INFO - Processing request of type ListToolsRequest
2025-05-30 16:00:49,404 - INFO - HTTP Request: POST http://localhost:8000/mcp "HTTP/1.1 307 Temporary Redirect"
2025-05-30 16:00:49,407 - INFO - HTTP Request: POST http://localhost:8000/mcp/ "HTTP/1.1 200 OK"
2025-05-30 16:00:49,414 - INFO - HTTP Request: POST http://localhost:8001/messages/?session_id=2fcb20d7741e489d860dbafb54803c0b "HTTP/1.1 202 Accepted"
2025-05-30 16:00:49,417 - INFO - Executing web_search...
2025-05-30 16:00:49,420 - INFO - HTTP Request: POST http://localhost:8001/messages/?session_id=2fcb20d7741e489d860dbafb54803c0b "HTTP/1.1 202 Accepted"
2025-05-30 16:00:55,865 - INFO - HTTP Request: POST https://api.together.xyz/v1/chat/completions "HTTP/1.1 200 OK"
2025-05-30 16:00:55,874 - INFO - 
Final response: 北京有许多著名的景点，这里有几个推荐给你：

1. 长城（AAAAA级景区）
2. 天安门广场（AAAAA级景区）
3. 北京动物园
4. 北京植物园
5. 北京朝阳公园
6. 北京朝阳九棵树

此外，还有其他一些著名景点，如天坛、颐和园、故宫、玉渊潭公园、EPSON teamLab无界空间等。这些地方各有特色，值得一游。例如，天安门广场是国家政治中心的重要标志；长城则以其悠久的历史和壮丽的景色闻名。希望这些信息对你有所帮助！
You: 请抓取https://www.datetoday.net/
2025-05-30 16:04:21,631 - INFO - HTTP Request: POST https://api.together.xyz/v1/chat/completions "HTTP/1.1 200 OK"
2025-05-30 16:04:21,639 - INFO - 
Assistant: {
    "tool": "fetch_website",
    "arguments": {
        "url": "https://www.datetoday.net/"
    }
}
2025-05-30 16:04:21,639 - INFO - Executing tool: fetch_website
2025-05-30 16:04:21,640 - INFO - With arguments: {'url': 'https://www.datetoday.net/'}
2025-05-30 16:04:21,642 - mcp.server.lowlevel.server - INFO - Processing request of type ListToolsRequest
2025-05-30 16:04:21,645 - INFO - Executing fetch_website...
2025-05-30 16:04:21,647 - mcp.server.lowlevel.server - INFO - Processing request of type CallToolRequest
2025-05-30 16:04:21,647 - root - INFO - functions.dict()={'fetch_website': <function fetch_website at 0x1101c77e0>, 'state_send_notification': <function state_send_notification at 0x1101c7a60>, 'stateless_send_notification': <function stateless_send_notification at 0x1101c7ba0>}
2025-05-30 16:04:22,919 - httpx - INFO - HTTP Request: GET https://www.datetoday.net/ "HTTP/1.1 200 OK"
2025-05-30 16:04:25,806 - INFO - HTTP Request: POST https://api.together.xyz/v1/chat/completions "HTTP/1.1 200 OK"
2025-05-30 16:04:25,816 - INFO - 
Final response: 今天是2025年5月30日，星期五。以下是今天日期的一些信息：

- **日期**：2025年5月30日
- **星期**：星期五
- **星期几**：第22周
- **日期**：第150天
- **月份数**：第5个月

今天的具体时间和格式如下：

- **今天的日期**（ISO 8601格式）：2025-05-30
- **今天的日期**（美国格式）：05/30/2025
- **今天的日期**（欧洲格式）：30/05/2025

此外，你还可以查看今天的具体时间戳和其他相关信息，如农历日期、月亮相位等。这个网站提供了非常详细的信息，帮助你了解今天的日期和时间。

如果你还有其他问题或需要更多信息，请告诉我！
You: quit
2025-05-30 16:04:53,535 - INFO - 
Exiting...
2025-05-30 16:04:53,544 - INFO - HTTP Request: DELETE http://localhost:8000/mcp "HTTP/1.1 307 Temporary Redirect"
2025-05-30 16:04:53,546 - INFO - HTTP Request: DELETE http://localhost:8000/mcp/ "HTTP/1.1 200 OK"
```