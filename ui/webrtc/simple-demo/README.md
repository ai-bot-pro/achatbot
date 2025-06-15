# simple webrtc demo intro
- this demo is test webrtc p2p bot audio stream with http signaling server, 
- use webrtc protocol(RTP) to send multimodal data(audio).

# run demo
0. download asr model
```shell
huggingface-cli download FunAudioLLM/SenseVoiceSmall  --local-dir ./models/FunAudioLLM/SenseVoiceSmall 
```
1. run signaling bot server:
- local start bot
```shell
python -m src.cmd.webrtc.signaling_bot_server -f config/bots/small_webrtc_server_bot.json
```
config/bots/websocket_server_bot.json
```json
{
  "chat_bot_name": "SmallWebrtcBot",
  "transport_type": "small_webrtc",
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

2. run webrtc voice agent web demo
```shell
cd ui/webrtc/simple-demo && python -m http.server
```
access http://localhost:8000/  to click `Connect` to chat with bot


# references
- https://github.com/ai-bot-pro/achatbot/pull/158
- [WebRTC Samples](https://webrtc.github.io/samples)
- [Web-API Docs](https://developer.mozilla.org/en-US/docs/Web/API)