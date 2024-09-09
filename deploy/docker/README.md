# Deploy APP

## fastapi-daily-chat-bot

### CPU
```
# build achatbot base img
make docker_cpu_debian_img
# build fastapi_daily_chat_bot base img 
make docker_cpu_debian_fastapi_daily_bot_img

# based ${A_ACHATBOT_IMG_TAG} img to build run img
# install all vad,asr,llm,tts dependency
make docker_cpu_debian_fastapi_daily_bot_run_img

# install vad,api asr,api llm,api tts dependency
make docker_cpu_debian_fastapi_daily_bot_run_api_models_img

# run container with Dockfile file
make docker_cpu_debian_fastapi_daily_bot_container_run

# run container with docker-compose.yaml file
make docker_cpu_debian_fastapi_daily_bot_containers_compose_run 

```

### GPU

> [!TODO]
