# webrtc+websocket lam audio2expression avatar bot demo intro
- this demo is test webrtc p2p bot audio stream with http signaling server, 
- use webrtc protocol(RTP) to send multimodal data(audio).
- use websocket protocol to get message(protobuff serialization) (audio,audio_expression).

# run demo
0. download asr model
```shell
#huggingface-cli download FunAudioLLM/SenseVoiceSmall  --local-dir ./models/FunAudioLLM/SenseVoiceSmall 

wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/LAM_audio2exp_streaming.tar -P ./models/LAM_audio2exp/
tar -xzvf ./models/LAM_audio2exp/LAM_audio2exp_streaming.tar -C ./models/LAM_audio2exp && rm ./models/LAM_audio2exp/LAM_audio2exp_streaming.tar
git clone --depth 1 https://www.modelscope.cn/AI-ModelScope/wav2vec2-base-960h.git ./models/facebook/wav2vec2-base-960h

```
1. run small_webrtc_fastapi_websocket_avatar_echo_bot server:
- local start bot
```shell
python -m src.cmd.webrtc_websocket.fastapi_ws_signaling_bot_serve -f config/bots/small_webrtc_fastapi_websocket_avatar_echo_bot.json
```
config/bots/small_webrtc_fastapi_websocket_avatar_echo_bot.json
```json
{
    "chat_bot_name": "SmallWebRTCFastapiWebsocketAvatarEchoBot",
    "config": {
        "avatar": {
            "args": {
                "audio_sample_rate": 16000,
                "wav2vec_dir": "./models/facebook/wav2vec2-base-960h",
                "weight_path": "./models/LAM_audio2exp/pretrained_models/lam_audio2exp_streaming.tar"
            },
            "tag": "lam_audio2expression_avatar"
        },
        "vad": {
            "args": {
                "stop_secs": 0.7
            },
            "tag": "silero_vad_analyzer"
        }
    },
    "config_list": [],
    "handle_sigint": false,
    "services": {
        "avatar": "lam_audio2expression_avatar",
        "pipeline": "achatbot",
        "vad": "silero"
    }
}
```

2. run webrtc + websocket voice avatar agent web demo
```shell
cd ui/webrtc_websocket/lam_audio2expression_avatar && python -m http.server
```
access http://localhost:8000/  to click `Connect` to chat with bot


# references
- https://github.com/ai-bot-pro/achatbot/pull/164
- [WebRTC Samples](https://webrtc.github.io/samples)
- [Websocket](https://en.wikipedia.org/wiki/WebSocket)
- [protobuf.js](https://github.com/protobufjs/protobuf.js)
- [Protocol Buffers Documentation](https://protobuf.dev/overview/)
- [Web-API Docs](https://developer.mozilla.org/en-US/docs/Web/API)