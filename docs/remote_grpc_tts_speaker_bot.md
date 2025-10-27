## Run remote grpc tts speaker bot

1. run `pip install "achatbot[remote_grpc_tts_server]"` to install dependencies to run grpc tts speaker bot server; 

```
ACHATBOT_PKG=1 python -m achatbot.cmd.grpc.speaker.server.serve
```

2. run `pip install "achatbot[remote_grpc_tts_client]"` to install dependencies to run grpc tts speaker bot client; 

```
ACHATBOT_PKG=1 TTS_TAG=tts_edge IS_RELOAD=1 python -m src.cmd.grpc.speaker.client
ACHATBOT_PKG=1 TTS_TAG=tts_g IS_RELOAD=1 python -m src.cmd.grpc.speaker.client
ACHATBOT_PKG=1 TTS_TAG=tts_coqui IS_RELOAD=1 python -m src.cmd.grpc.speaker.client
ACHATBOT_PKG=1 TTS_TAG=tts_chat IS_RELOAD=1 python -m src.cmd.grpc.speaker.client
ACHATBOT_PKG=1 TTS_TAG=tts_cosy_voice IS_RELOAD=1 python -m src.cmd.grpc.speaker.client
ACHATBOT_PKG=1 TTS_TAG=tts_fishspeech IS_RELOAD=1 python -m src.cmd.grpc.speaker.client
ACHATBOT_PKG=1 TTS_TAG=tts_f5 IS_RELOAD=1 python -m src.cmd.grpc.speaker.client
ACHATBOT_PKG=1 TTS_TAG=tts_openvoicev2 IS_RELOAD=1 python -m src.cmd.grpc.speaker.client
ACHATBOT_PKG=1 TTS_TAG=tts_kokoro IS_RELOAD=1 python -m src.cmd.grpc.speaker.client
ACHATBOT_PKG=1 TTS_TAG=tts_onnx_kokoro IS_RELOAD=1 KOKORO_ESPEAK_NG_LIB_PATH=/usr/local/lib/libespeak-ng.1.dylib KOKORO_LANGUAGE=cmn python -m src.cmd.grpc.speaker.client
ACHATBOT_PKG=1 TTS_TAG=tts_cosy_voice2 \
    COSY_VOICE_MODELS_DIR=./models/FunAudioLLM/CosyVoice2-0.5B \
    COSY_VOICE_REFERENCE_AUDIO_PATH=./test/audio_files/asr_example_zh.wav \
    IS_RELOAD=1 python -m src.cmd.grpc.speaker.client
```

