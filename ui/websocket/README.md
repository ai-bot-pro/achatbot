# simple websocket demo intro
- this demo is test webscoket server bot audio stream, 
- use protobuf to encode and decode data, 
- use websocket protocol to send data.

# bots:
- voice live chat bot
- asr live bot
- asr translate tts bot

# run demo
1. run websocket server bot with 2 ways:
- local start bot
```shell
# voice live chat bot
python -m src.cmd.bots.main -f config/bots/websocket_server_bot.json

# asr live bot
python -m src.cmd.websocket.server.fastapi_ws_bot_serve -f config/bots/fastapi_websocket_asr_live_bot.json

# asr translate tts bot
python -m src.cmd.websocket.server.fastapi_ws_bot_serve -f config/bots/fastapi_websocket_asr_translate_tts_bot.json
```
- config/bots/websocket_server_bot.json
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
- config/bots/fastapi_websocket_asr_live_bot.json 
```json
{
  "chat_bot_name": "FastapiWebsocketStreamingASRBot",
  "transport_type": "websocket",
  "handle_sigint": false,
  "services": {
    "pipeline": "achatbot",
    "vad": "silero",
    "asr": "asr_streaming_sensevoice",
    "punctuation":"punc_ct_tranformer"
  },
  "config": {
    "vad": {
      "tag": "silero_vad_analyzer",
      "args": {
        "start_secs": 0.032,
        "stop_secs": 0.32,
        "confidence": 0.7,
        "min_volume": 0.4
      }
    },
    "asr": {
      "tag": "asr_streaming_sensevoice",
      "args": {
        "chunk_size": 10,
        "padding": 8,
        "beam_size": 3,
        "contexts": [],
        "language": "zh",
        "textnorm": false,
        "model": "./models/FunAudioLLM/SenseVoiceSmall"
      }
    },
    "punctuation": {
      "tag": "punc_ct_tranformer",
      "args": {
        "model":"./models/iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727"
      }
    }
  },
  "config_list": []
}
```
- config/bots/fastapi_websocket_asr_translate_tts_bot.json
```json
{
  "chat_bot_name": "FastapiWebsocketServerASRTranslateTTSBot",
  "transport_type": "websocket",
  "handle_sigint": false,
  "services": {
    "pipeline": "achatbot",
    "vad": "silero",
    "asr": "sense_voice",
    "punctuation": "punc_ct_tranformer",
    "translate_llm": "llm_llamacpp_generator",
    "tts": "edge"
  },
  "config": {
    "vad": {
      "tag": "silero_vad_analyzer",
      "args": {
        "start_secs": 0.032,
        "stop_secs": 0.32,
        "confidence": 0.7,
        "min_volume": 0.6,
        "onnx": true
      }
    },
    "asr": {
      "tag": "sense_voice_asr",
      "args": {
        "language": "zn",
        "model_name_or_path": "./models/FunAudioLLM/SenseVoiceSmall"
      }
    },
    "punctuation": {
      "tag": "punc_ct_tranformer_onnx_offline",
      "args": {
        "model": "./models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
      }
    },
    "translate_llm": {
      "init_prompt":"hi, welcome to speak with translation bot.",
      "model": "./models/ByteDance-Seed/Seed-X-PPO-7B",
      "src": "zh",
      "target": "en",
      "streaming": false,
      "tag": "llm_llamacpp_generator",
      "args": {
        "save_chat_history": false,
        "model_path": "./models/Seed-X-PPO-7B.Q2_K.gguf",
        "model_type": "generate",
        "llm_temperature": 0.0,
        "llm_stop_ids": [2],
        "llm_max_tokens": 2048
      }
    },
    "tts": {
      "aggregate_sentences": false,
      "push_text_frames": true,
      "remove_punctuation": false,
      "tag": "tts_edge",
      "args": {
        "voice_name": "en-US-GuyNeural",
        "language": "en",
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

# curl start voice chat bot api
curl -XPOST "http://0.0.0.0:4321/bot_join/WebsocketServerBot" \                                     
    -H "Content-Type: application/json" \
    -d $'{"transport_type":"websocket","config":{"asr": { "tag": "sense_voice_asr", "args": { "language": "zn", "model_name_or_path": "./models/FunAudioLLM/SenseVoiceSmall" } },"llm": { "tag": "openai_llm_processor", "base_url": "https://api.together.xyz/v1", "model": "Qwen/Qwen2-72B-Instruct", "language": "zh", "messages": [ { "role": "system", "content": "你是一名叫奥利给的智能助理。保持回答简短和清晰。请用中文回答。" } ] }, "tts": { "tag": "tts_edge", "args": { "voice_name": "zh-CN-YunjianNeural", "language": "zh", "gender": "Male" } } }}'  | jq .
```

2. run websocket client 
```shell
cd ui/websocket && python -m http.server
```
- access http://localhost:8000/simple-demo  to click `Start Audio` to chat with Voice bot
- access http://localhost:8000/asr_live  to click `Start Audio` to transcript with ASR Live bot
- access http://localhost:8000/translation  to click `Start Audio` to speech translation with Translation bot


> [!NOTE]
> - [data_frames.proto](./protos/data_frames.proto) text/image/audio pb schema from https://github.com/ai-bot-pro/pipeline-py/blob/main/apipeline/frames/data_frames.proto 
> - [asr_data_frames.proto](./protos/asr_data_frames.proto) Frame seq id don't repeat [data_frames.proto](./protos/data_frames.proto)

# references
- [Websocket](https://en.wikipedia.org/wiki/WebSocket)
- [protobuf.js](https://github.com/protobufjs/protobuf.js)
- [Protocol Buffers Documentation](https://protobuf.dev/overview/)
- [Web-API Docs](https://developer.mozilla.org/en-US/docs/Web/API)