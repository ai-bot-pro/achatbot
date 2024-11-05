# simple websocket demo intro
- this demo is test webscoket server bot audio stream, 
- use protobuf to encode and decode data, 
- use websocket protocol to send data.

# run demo
1. run websocket server bot with 2 ways:
- local start bot
```shell
python -m src.cmd.bots.main -f config/bots/websocket_server_bot.json
```
config/bots/websocket_server_bot.json
```json
{
  "chat_bot_name": "WebsocketServerBot",
  "transport_type": "websocket",
  "websocket_server_port": 8765,
  "websocket_server_host": "localhost",
  "services": {
    "pipeline": "achatbot",
    "vad": "silero",
    "asr": "sense_voice",
    "llm": "together",
    "tts": "edge"
  },
  "config": {
    "vad": {
      "tag": "silero_vad_analyzer",
      "args": { "stop_secs": 0.7 }
    },
    "asr": {
      "tag": "sense_voice_asr",
      "args": {
        "language": "zn",
        "model_name_or_path": "./models/FunAudioLLM/SenseVoiceSmall"
      }
    },
    "llm": {
      "tag": "openai_llm_processor",
      "base_url": "https://api.together.xyz/v1",
      "model": "Qwen/Qwen2-72B-Instruct",
      "language": "zh",
      "messages": [
        {
          "role": "system",
          "content": "你是一名叫奥利给的智能助理。保持回答简短和清晰。请用中文回答。"
        }
      ]
    },
    "tts": {
      "tag": "tts_edge",
      "args": {
        "voice_name": "zh-CN-YunjianNeural",
        "language": "zh",
        "gender": "Male"
      }
    }
  },
  "config_list": []
}
```

- http api start bot
```shell
# run http server
python -m src.cmd.http.server.fastapi_daily_bot_serve 

# curl start bot api
curl -XPOST "http://0.0.0.0:4321/bot_join/WebsocketServerBot" \                                     
    -H "Content-Type: application/json" \
    -d $'{"transport_type":"websocket","config":{"asr": { "tag": "sense_voice_asr", "args": { "language": "zn", "model_name_or_path": "./models/FunAudioLLM/SenseVoiceSmall" } },"llm": { "tag": "openai_llm_processor", "base_url": "https://api.together.xyz/v1", "model": "Qwen/Qwen2-72B-Instruct", "language": "zh", "messages": [ { "role": "system", "content": "你是一名叫奥利给的智能助理。保持回答简短和清晰。请用中文回答。" } ] }, "tts": { "tag": "tts_edge", "args": { "voice_name": "zh-CN-YunjianNeural", "language": "zh", "gender": "Male" } } }}'  | jq .
```

2. run websocket client 
```shell
cd ui/websocket/simple-demo && python -m http.server
```
access http://localhost:8000/  to click `Start Audio` to chat with bot


> [!NOTE]: [frames.proto](./frames.proto) text/image/audio pb schema from https://github.com/ai-bot-pro/pipeline-py/blob/main/apipeline/frames/data_frames.proto 

# references
- [Websocket](https://en.wikipedia.org/wiki/WebSocket)
- [protobuf.js](https://github.com/protobufjs/protobuf.js)
- [Protocol Buffers Documentation](https://protobuf.dev/overview/)
- [Web-API Docs](https://developer.mozilla.org/en-US/docs/Web/API)