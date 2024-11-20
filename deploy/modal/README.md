# intro
the Starter plan with $30.00 included compute credits per month, for development, so nice~

# modal
> [!NOTE] in deploy/modal dir to run shell
```shell
pip install modal 

modal setup

# create worksapce achatbot environment
modal environment create achatbot

# Created Volume 'bot_config' in environment 'achatbot'
# for remote local bot config
modal volume create -e achatbot bot_config
```

## modal run 
- run serverless remote function call like local to dev, so nice~ (auto schedule)
- crob jobs to run schedule task

## modal serve
- use fastapi(ASGI modal app) to run serverless web service
- web server endpoint: `https://{profile}-{environment}--{app_name}-{class_name}-app-dev.modal.run` (if app_name + class_name is so long, hash tag name to replace)
```shell
# serve
modal serve -e achatbot src/fastapi_serve.py

# bot serve e.g.:
# webrtc_audio_bot serve on default pip image
# need create .env.example to modal Secrets
IMAGE_NAME=default modal serve -e achatbot src/fastapi_webrtc_audio_bot_serve.py

# put local config to modal volume bot_config / dir
modal volume put -e achatbot bot_config  ../../config/bots/fastapi_websocket_moshi_voice_bot.json / -f
# ws_moshi_voice_bot serve on default pip image
IMAGE_NAME=default modal serve -e achatbot src/fastapi_ws_moshi_voice_bot_serve.py
# run moshi_opus_stream_ws_pb_client to chat with moshi in CLI
python -m achatbot.cmd.websocket.moshi_opus_stream_ws_pb_client --endpoint https://weedge-achatbot--fastapi-ws-moshi-voice-bot-srv-app-dev.modal.run/
```

## modal deploy 


# references (nice docs) 👍
- https://modal.com/docs/guide
- https://modal.com/docs/guide/gpu
- https://modal.com/docs/guide/cuda
- https://modal.com/docs/guide/volumes
- https://modal.com/docs/examples/basic_web
- https://modal.com/docs/guide/cron
- https://github.com/modal-labs/modal-examples