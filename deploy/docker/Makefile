# --------
docker_cpu_debian_img:
	DOCKER_HOST=unix://$(HOME)/.docker/run/docker.sock docker build \
		-t achatbot:base \
		--build-arg A_PYTHON_IMG_TAG=3.11-slim \
		--build-arg A_ACHATBOT_DEPENDENCY=achatbot \
		--no-cache \
		-f Dockerfile.cpu.debian .

docker_cpu_debian_fastapi_daily_bot_img:
	DOCKER_HOST=unix://$(HOME)/.docker/run/docker.sock docker build \
		-t achatbot:fastapi_daily_bot \
		--build-arg A_PYTHON_IMG_TAG=3.11-slim \
		--build-arg A_ACHATBOT_DEPENDENCY="achatbot[fastapi_bot_server, daily_rtvi_bot, daily_langchain_rag_bot, speech_vad_analyzer, asr_processor, llm_processor, tts_processor]" \
		--no-cache \
		-f Dockerfile.cpu.debian .

# install all vad,asr,llm,tts dependency
docker_cpu_debian_fastapi_daily_bot_run_img:
	DOCKER_HOST=unix://$(HOME)/.docker/run/docker.sock docker build \
		-t achatbot:fastapi_daily_bot_run_all \
		--build-arg A_ACHATBOT_IMG_TAG=base \
		--build-arg A_ACHATBOT_DEPENDENCY="achatbot[fastapi_bot_server, daily_rtvi_bot, daily_langchain_rag_bot, speech_vad_analyzer, asr_processor, llm_processor, tts_processor]" \
		--no-cache \
		-f fastapi-daily-chat-bot/Dockerfile.cpu.debian .

# install vad,api asr,api llm,api tts dependency
docker_cpu_debian_fastapi_daily_bot_run_api_models_img:
	DOCKER_HOST=unix://$(HOME)/.docker/run/docker.sock docker build \
		-t achatbot:fastapi_daily_bot_run \
		--build-arg A_ACHATBOT_IMG_TAG=base \
		--build-arg A_ACHATBOT_DEPENDENCY="achatbot[fastapi_bot_server, daily_rtvi_bot, daily_langchain_rag_bot, speech_vad_analyzer, deepgram_asr_processor, openai, tts_edge]" \
		--no-cache \
		-f fastapi-daily-chat-bot/Dockerfile.cpu.debian .


#---- run container ----

docker_cpu_debian_fastapi_daily_bot_container_run:
	DOCKER_HOST=unix://$(HOME)/.docker/run/docker.sock docker run \
		-it \
		--name achatbot_fastapi_daily_bot \
		-p 8000:8000 \
		--env-file .env \
		achatbot:fastapi_daily_bot_run

docker_cpu_debian_fastapi_daily_bot_containers_compose_run:
	docker-compose \
		-f fastapi-daily-chat-bot/docker-compose.yaml up
