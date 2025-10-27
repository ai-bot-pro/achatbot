## Run remote rpc chat bot worker

1. run `pip install "achatbot[remote_rpc_chat_bot_be_worker]"` to install dependencies to run rpc chat bot BE worker; e.g. :
   - use dufault env params to run rpc chat bot BE worker

```
ACHATBOT_PKG=1 RUN_OP=be TQDM_DISABLE=True \
    TTS_TAG=tts_edge \
    python -m achatbot.cmd.grpc.terminal-chat.generate_audio2audio > ~/.achatbot/log/be_std_out.log
```

2. run `pip install "achatbot[remote_rpc_chat_bot_fe]"` to install dependencies to run rpc chat bot FE; 

```
ACHATBOT_PKG=1 RUN_OP=fe \
    TTS_TAG=tts_edge \
    python -m achatbot.cmd.grpc.terminal-chat.generate_audio2audio > ~/.achatbot/log/fe_std_out.log
```

